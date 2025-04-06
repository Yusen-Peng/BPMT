import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import collate_fn_batch_padding
from first_phase_baseline import BaseT1, mask_keypoints
from second_phase_baseline import BaseT2, CrossAttention, load_T1


def load_T2(model_path: str, out_dim_A: int, out_dim_B: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT2:
    """
        loads a BaseT2 model from a checkpoint
    """
    model = BaseT2(out_dim_A=out_dim_A, out_dim_B=out_dim_B, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # optionally freeze the model parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        
    # move model to device and return the model
    return model.to(device)



class GaitRecognitionHead(nn.Module):
    """
        A simple linear head for gait recognition.
        The model consists of a linear layer that maps the output of the transformer to the number of classes.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)



def finetuning(
    dataloader,
    t1_map,
    t2_map,
    cross_attn,
    gait_head,
    optimizer,
    num_epochs=100,
    freeze=True,
    device='cuda'
):
    
    # we are training cross-attention and gait recognition head
    cross_attn.train()
    gait_head.train()

    # we can freeze T1 and T2 parameters
    if freeze:  
        for m in t1_map.values():
            m.eval()
        for t2 in t2_map.values():
            t2.eval()

    criterion = nn.CrossEntropyLoss()

    modalities = ['torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_samples = 0

        for batch in dataloader:
            # Suppose each batch is (torso_seq, la_seq, ra_seq, ll_seq, rl_seq, labels)
            # all shape [B, T, 2*num_joints], except labels => [B]
            (torso_seq, la_seq, ra_seq, ll_seq, rl_seq, labels) = batch
            torso_seq = torso_seq.to(device)
            la_seq    = la_seq.to(device)
            ra_seq    = ra_seq.to(device)
            ll_seq    = ll_seq.to(device)
            rl_seq    = rl_seq.to(device)
            labels    = labels.to(device)

            # A dict to hold final representations for each modality, e.g. final_reprs['torso'] = [B, d_model]
            final_reprs = {}

            # For convenience, store the raw sequences in a dict as well
            seqs = {
                'torso': torso_seq,
                'left_arm': la_seq,
                'right_arm': ra_seq,
                'left_leg': ll_seq,
                'right_leg': rl_seq
            }

            # 1) For each modality M, get T1 encoding
            encoded_t1 = {}
            for M in modalities:
                encoded_t1[M] = t1_map[M].encode(seqs[M])
                # shape [B, T_M, d_model]

            # 2) For each modality M, compute T2 encoding with each other modality O
            #    i.e., "encoded_M_with_O" 
            #    Then cross-attend: cross_attn(encoded_t1[M], encoded_M_with_O, encoded_M_with_O).
            #    Collect the cross-attention outputs in a list for later fusion.
            for M in modalities:
                cross_outputs_M = []  # will collect cross_out_M_O for each O != M

                for O in modalities:
                    if O == M:
                        continue
                    # We must figure out the order to feed T2. We trained T2 on (M, O).
                    # Ensure we retrieve the T2 in a consistent (M, O) or (O, M) order
                    pair_key = tuple(sorted([M, O]))

                    # 2A) concat raw sequences => feed T2 => we only want the portion for M
                    cat_seq = torch.cat([seqs[M], seqs[O]], dim=1)
                    # shape: [B, T_M + T_O, d_in]

                    # We'll figure out which portion is "A" vs. "B" in T2
                    # If pair_key = (M, O), then T_A = T_M. If pair_key=(O, M), T_A=T_O
                    # So let's define:
                    if pair_key == (M, O):
                        T_A = seqs[M].size(1)
                        encodedA, encodedB = t2_map[pair_key].encode(cat_seq, T_A=T_A)
                        encoded_M_with_O = encodedA  # the first chunk is M
                    else:
                        # pair_key == (O, M)
                        T_A = seqs[O].size(1)
                        encodedA, encodedB = t2_map[pair_key].encode(cat_seq, T_A=T_A)
                        encoded_M_with_O = encodedB  # the second chunk is M

                    # 2B) cross attention
                    # Q=encoded_t1[M], K=encoded_M_with_O, V=encoded_M_with_O
                    cross_out = cross_attn(encoded_t1[M], encoded_M_with_O, encoded_M_with_O)
                    cross_outputs_M.append(cross_out)

                # 3) fuse cross_outputs_M (which is a list of length 4 in the torso example)
                #    shape for each cross_out is [B, T_M, d_model]
                #    simplest approach: sum or average
                cross_sum = sum(cross_outputs_M)  # elementwise sum
                cross_fused = cross_sum / len(cross_outputs_M)
                # shape [B, T_M, d_model]

                # 4) optionally pool over time => [B, d_model]
                final_reprs[M] = cross_fused.mean(dim=1)

            # 5) fuse across all 5 modalities => final
            # each final_reprs[M] is [B, d_model]
            fused_all = torch.cat([
                final_reprs['torso'],
                final_reprs['left_arm'],
                final_reprs['right_arm'],
                final_reprs['left_leg'],
                final_reprs['right_leg']
            ], dim=-1)  # shape [B, 5*d_model]

            # 6) classification
            logits = gait_head(fused_all)  # => [B, num_classes]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / (total_samples + 1e-9)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    print("Finetuning complete!")











