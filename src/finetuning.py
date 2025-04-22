import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import collate_fn_batch_padding
from first_phase_baseline import BaseT1, mask_keypoints
from second_phase_baseline import BaseT2, CrossAttention, load_T1


def load_cross_attn(path: str,
                    d_model: int = 128,
                    nhead: int = 4,
                    device: str = "cuda") -> CrossAttention:
    """
        loads a CrossAttention model from a checkpoint
    """
    layer = CrossAttention(d_model=d_model, nhead=nhead)
    layer.load_state_dict(torch.load(path, map_location="cpu"))
    return layer.to(device)

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



def T1_encoding(t1_map: dict[str, BaseT1], torso_seq, la_seq, ra_seq, ll_seq, rl_seq):
    """
        get T1 encoding for each modality.
    """

    encoded_t1_torso = t1_map['Torso'].encode(torso_seq)
    encoded_t1_left_arm = t1_map['Left_Arm'].encode(la_seq)
    encoded_t1_right_arm = t1_map['Right_Arm'].encode(ra_seq)
    encoded_t1_left_leg = t1_map['Left_Leg'].encode(ll_seq)
    encoded_t1_right_leg = t1_map['Right_Leg'].encode(rl_seq)

    return {
        'torso': encoded_t1_torso,
        'left_arm': encoded_t1_left_arm,
        'right_arm': encoded_t1_right_arm,
        'left_leg': encoded_t1_left_leg,
        'right_leg': encoded_t1_right_leg
    }


def T2_encoding(t2_map: dict[str, BaseT2], encoded_t1_torso: torch.Tensor, encoded_t1_left_arm: torch.Tensor, encoded_t1_right_arm: torch.Tensor, encoded_t1_left_leg: torch.Tensor, encoded_t1_right_leg: torch.Tensor, cross_attn_modules_before_T2: dict[str, CrossAttention]):
    """
        get T2 encoding for each modality.
    """
    # 1. torso - left arm
    cross_attn_torso_left_arm = cross_attn_modules_before_T2['torso_left_arm']
    torso_attends_left_arm = cross_attn_torso_left_arm(encoded_t1_torso, encoded_t1_left_arm, encoded_t1_left_arm)
    left_arm_attends_torso = cross_attn_torso_left_arm(encoded_t1_left_arm, encoded_t1_torso, encoded_t1_torso)

    torso_concat_left_arm = torch.cat([torso_attends_left_arm, left_arm_attends_torso], dim=1)
    torso_recons_with_left_arm, left_arm_recons_with_torso = t2_map['torso_left_arm'].encode(torso_concat_left_arm, T_A=encoded_t1_torso.size(1))

    # 2. torso - right arm
    cross_attn_torso_right_arm = cross_attn_modules_before_T2['torso_right_arm']
    torso_attends_right_arm = cross_attn_torso_right_arm(encoded_t1_torso, encoded_t1_right_arm, encoded_t1_right_arm)
    right_arm_attends_torso = cross_attn_torso_right_arm(encoded_t1_right_arm, encoded_t1_torso, encoded_t1_torso)
    torso_concat_right_arm = torch.cat([torso_attends_right_arm, right_arm_attends_torso], dim=1)
    torso_recons_with_right_arm, right_arm_recons_with_torso = t2_map['torso_right_arm'].encode(torso_concat_right_arm, T_A=encoded_t1_torso.size(1))

    # 3. torso - left leg
    cross_attn_torso_left_leg = cross_attn_modules_before_T2['torso_left_leg']
    torso_attends_left_leg = cross_attn_torso_left_leg(encoded_t1_torso, encoded_t1_left_leg, encoded_t1_left_leg)
    left_leg_attends_torso = cross_attn_torso_left_leg(encoded_t1_left_leg, encoded_t1_torso, encoded_t1_torso)
    torso_concat_left_leg = torch.cat([torso_attends_left_leg, left_leg_attends_torso], dim=1)
    torso_recons_with_left_leg, left_leg_recons_with_torso = t2_map['torso_left_leg'].encode(torso_concat_left_leg, T_A=encoded_t1_torso.size(1))

    # 4. torso - right leg
    cross_attn_torso_right_leg = cross_attn_modules_before_T2['torso_right_leg']
    torso_attends_right_leg = cross_attn_torso_right_leg(encoded_t1_torso, encoded_t1_right_leg, encoded_t1_right_leg)
    right_leg_attends_torso = cross_attn_torso_right_leg(encoded_t1_right_leg, encoded_t1_torso, encoded_t1_torso)
    torso_concat_right_leg = torch.cat([torso_attends_right_leg, right_leg_attends_torso], dim=1)
    torso_recons_with_right_leg, right_leg_recons_with_torso = t2_map['torso_right_leg'].encode(torso_concat_right_leg, T_A=encoded_t1_torso.size(1))

    # 5. left arm - right arm
    cross_attn_left_arm_right_arm = cross_attn_modules_before_T2['left_arm_right_arm']
    left_arm_attends_right_arm = cross_attn_left_arm_right_arm(encoded_t1_left_arm, encoded_t1_right_arm, encoded_t1_right_arm)
    right_arm_attends_left_arm = cross_attn_left_arm_right_arm(encoded_t1_right_arm, encoded_t1_left_arm, encoded_t1_left_arm)
    left_arm_concat_right_arm = torch.cat([left_arm_attends_right_arm, right_arm_attends_left_arm], dim=1)
    left_arm_recons_with_right_arm, right_arm_recons_with_left_arm = t2_map['left_arm_right_arm'].encode(left_arm_concat_right_arm, T_A=encoded_t1_left_arm.size(1))

    # 6. left arm - left leg
    cross_attn_left_arm_left_leg = cross_attn_modules_before_T2['left_arm_left_leg']
    left_arm_attends_left_leg = cross_attn_left_arm_left_leg(encoded_t1_left_arm, encoded_t1_left_leg, encoded_t1_left_leg)
    left_leg_attends_left_arm = cross_attn_left_arm_left_leg(encoded_t1_left_leg, encoded_t1_left_arm, encoded_t1_left_arm)
    left_arm_concat_left_leg = torch.cat([left_arm_attends_left_leg, left_leg_attends_left_arm], dim=1)
    left_arm_recons_with_left_leg, left_leg_recons_with_left_arm = t2_map['left_arm_left_leg'].encode(left_arm_concat_left_leg, T_A=encoded_t1_left_arm.size(1))
    
    # 7. left arm - right leg
    cross_attn_left_arm_right_leg = cross_attn_modules_before_T2['left_arm_right_leg']
    left_arm_attends_right_leg = cross_attn_left_arm_right_leg(encoded_t1_left_arm, encoded_t1_right_leg, encoded_t1_right_leg)
    right_leg_attends_left_arm = cross_attn_left_arm_right_leg(encoded_t1_right_leg, encoded_t1_left_arm, encoded_t1_left_arm)
    left_arm_concat_right_leg = torch.cat([left_arm_attends_right_leg, right_leg_attends_left_arm], dim=1)
    left_arm_recons_with_right_leg, right_leg_recons_with_left_arm = t2_map['left_arm_right_leg'].encode(left_arm_concat_right_leg, T_A=encoded_t1_left_arm.size(1))

    # 8. right arm - left leg
    cross_attn_right_arm_left_leg = cross_attn_modules_before_T2['right_arm_left_leg']
    right_arm_attends_left_leg = cross_attn_right_arm_left_leg(encoded_t1_right_arm, encoded_t1_left_leg, encoded_t1_left_leg)
    left_leg_attends_right_arm = cross_attn_right_arm_left_leg(encoded_t1_left_leg, encoded_t1_right_arm, encoded_t1_right_arm)
    right_arm_concat_left_leg = torch.cat([right_arm_attends_left_leg, left_leg_attends_right_arm], dim=1)
    right_arm_recons_with_left_leg, left_leg_recons_with_right_arm = t2_map['right_arm_left_leg'].encode(right_arm_concat_left_leg, T_A=encoded_t1_right_arm.size(1))

    # 9. right arm - right leg
    cross_attn_right_arm_right_leg = cross_attn_modules_before_T2['right_arm_right_leg']
    right_arm_attends_right_leg = cross_attn_right_arm_right_leg(encoded_t1_right_arm, encoded_t1_right_leg, encoded_t1_right_leg)
    right_leg_attends_right_arm = cross_attn_right_arm_right_leg(encoded_t1_right_leg, encoded_t1_right_arm, encoded_t1_right_arm)
    right_arm_concat_right_leg = torch.cat([right_arm_attends_right_leg, right_leg_attends_right_arm], dim=1)
    right_arm_recons_with_right_leg, right_leg_recons_with_right_arm = t2_map['right_arm_right_leg'].encode(right_arm_concat_right_leg, T_A=encoded_t1_right_arm.size(1))

    # 10. left leg - right leg
    cross_attn_left_leg_right_leg = cross_attn_modules_before_T2['left_leg_right_leg']
    left_leg_attends_right_leg = cross_attn_left_leg_right_leg(encoded_t1_left_leg, encoded_t1_right_leg, encoded_t1_right_leg)
    right_leg_attends_left_leg = cross_attn_left_leg_right_leg(encoded_t1_right_leg, encoded_t1_left_leg, encoded_t1_left_leg)
    left_leg_concat_right_leg = torch.cat([left_leg_attends_right_leg, right_leg_attends_left_leg], dim=1)
    left_leg_recons_with_right_leg, right_leg_recons_with_left_leg = t2_map['left_leg_right_leg'].encode(left_leg_concat_right_leg, T_A=encoded_t1_left_leg.size(1))


    # get T2 encoding list for each modality
    # torso, left_arm, right_arm, left_leg, right_leg
    return {
        'torso': [torso_recons_with_left_arm, torso_recons_with_right_arm, torso_recons_with_left_leg, torso_recons_with_right_leg],
        'left_arm': [left_arm_recons_with_torso, left_arm_recons_with_right_arm, left_arm_recons_with_left_leg, left_arm_recons_with_right_leg],
        'right_arm': [right_arm_recons_with_torso, right_arm_recons_with_left_arm, right_arm_recons_with_left_leg, right_arm_recons_with_right_leg],
        'left_leg': [left_leg_recons_with_torso, left_leg_recons_with_left_arm, left_leg_recons_with_right_arm, left_leg_recons_with_right_leg],
        'right_leg': [right_leg_recons_with_torso, right_leg_recons_with_left_arm, right_leg_recons_with_left_leg, right_leg_recons_with_right_arm]
    }


def finetuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    t1_map: dict[str, BaseT1],
    t2_map: dict[str, BaseT2],
    cross_attn_modules_before_T2: dict[str, CrossAttention],
    gait_head: GaitRecognitionHead,
    d_model=128,
    nhead=4,
    num_epochs=100,
    freeze=True,
    device='cuda'
):
    
    # cross-attention modeles after T2
    cross_attn_modules_after_T2 = {
        'torso': CrossAttention(d_model=d_model, nhead=nhead).to(device),
        'left_arm': CrossAttention(d_model=d_model, nhead=nhead).to(device),
        'right_arm': CrossAttention(d_model=d_model, nhead=nhead).to(device),
        'left_leg': CrossAttention(d_model=d_model, nhead=nhead).to(device),
        'right_leg': CrossAttention(d_model=d_model, nhead=nhead).to(device)
    }

    # we can freeze T1 and T2 parameters
    if freeze:  
        for m in t1_map.values():
            m.eval()
        for t2 in t2_map.values():
            t2.eval()


    gait_head.train()
    for m in cross_attn_modules_before_T2.values():
        m.train()
    for m in cross_attn_modules_after_T2.values():
        m.train()

    # use cross-entropy loss for gait recognition/classification
    criterion = nn.CrossEntropyLoss()

    # define the optimizer
    params = list(gait_head.parameters())
    params += [p for m in cross_attn_modules_before_T2.values() for p in m.parameters()]
    params += [p for m in cross_attn_modules_after_T2.values() for p in m.parameters()]


    # if we ever want to finetune T1 and T2, we need to add their parameters to the optimizer as well
    if not freeze:
        for m in t1_map.values():
            params += list(m.parameters())
        for m in t2_map.values():
            params += list(m.parameters())
    
    # save both train and val loss
    train_losses = []
    val_losses = []

    # IMPORTANT: use a small learning rate for finetuning
    # this is because the model is already trained and we want to finetune it
    # use weight decay for regularization
    # FIXME: tune the learning rate here ALWAYS
    optimizer = optim.Adam(params, lr=1e-5, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            # unpack the batch
            (torso_seq, la_seq, ra_seq, ll_seq, rl_seq, labels) = batch
            torso_seq = torso_seq.to(device)
            la_seq = la_seq.to(device)
            ra_seq = ra_seq.to(device)
            ll_seq = ll_seq.to(device)
            rl_seq = rl_seq.to(device)
            labels = labels.to(device)

            # get T1 encoding for each modality
            encoded_t1 = T1_encoding(t1_map, torso_seq, la_seq, ra_seq, ll_seq, rl_seq)
            encoded_t1_torso = encoded_t1['torso']
            encoded_t1_left_arm = encoded_t1['left_arm']
            encoded_t1_right_arm = encoded_t1['right_arm']
            encoded_t1_left_leg = encoded_t1['left_leg']
            encoded_t1_right_leg = encoded_t1['right_leg']

    
            # get T2 encoding with each other modality O for each modality M
            encoded_T2 = T2_encoding(t2_map, encoded_t1_torso, encoded_t1_left_arm, encoded_t1_right_arm, encoded_t1_left_leg, encoded_t1_right_leg, cross_attn_modules_before_T2)
            encoded_t2_torso = encoded_T2['torso']
            encoded_t2_left_arm = encoded_T2['left_arm']
            encoded_t2_right_arm = encoded_T2['right_arm']
            encoded_t2_left_leg = encoded_T2['left_leg']
            encoded_t2_right_leg = encoded_T2['right_leg']


            # for each modality, do cross-atttention between individual T1 encoding and T2 encoding list
            # here, we do joint cross-attention over the aggregation of 4 other encodings
            # torso
            Q = encoded_t1_torso
            K = V = torch.cat([
                encoded_t2_torso[0],
                encoded_t2_torso[1],
                encoded_t2_torso[2],
                encoded_t2_torso[3]
            ], dim=1)
            cross_out_torso = cross_attn_modules_after_T2['torso'](Q, K, V)

            # left arm
            Q = encoded_t1_left_arm
            K = V = torch.cat([
                encoded_t2_left_arm[0],
                encoded_t2_left_arm[1],
                encoded_t2_left_arm[2],
                encoded_t2_left_arm[3]
            ], dim=1)
            cross_out_left_arm = cross_attn_modules_after_T2['left_arm'](Q, K, V)

            # right arm
            Q = encoded_t1_right_arm
            K = V = torch.cat([
                encoded_t2_right_arm[0],
                encoded_t2_right_arm[1],
                encoded_t2_right_arm[2],
                encoded_t2_right_arm[3]
            ], dim=1)
            cross_out_right_arm = cross_attn_modules_after_T2['right_arm'](Q, K, V)
            
            # left leg
            Q = encoded_t1_left_leg
            K = V = torch.cat([
                encoded_t2_left_leg[0],
                encoded_t2_left_leg[1],
                encoded_t2_left_leg[2],
                encoded_t2_left_leg[3]
            ], dim=1)
            cross_out_left_leg = cross_attn_modules_after_T2['left_leg'](Q, K, V)

            # right leg
            Q = encoded_t1_right_leg
            K = V = torch.cat([
                encoded_t2_right_leg[0],
                encoded_t2_right_leg[1],
                encoded_t2_right_leg[2],
                encoded_t2_right_leg[3]
            ], dim=1)
            cross_out_right_leg = cross_attn_modules_after_T2['right_leg'](Q, K, V)

            # average pooling over time
            pooled_torso = cross_out_torso.mean(dim=1)
            pooled_left_arm = cross_out_left_arm.mean(dim=1)
            pooled_right_arm = cross_out_right_arm.mean(dim=1)
            pooled_left_leg = cross_out_left_leg.mean(dim=1)
            pooled_right_leg = cross_out_right_leg.mean(dim=1)

            # final fused representation
            final_reprs = torch.cat([
                pooled_torso,
                pooled_left_arm,
                pooled_right_arm,
                pooled_left_leg,
                pooled_right_leg
            ], dim=-1)

            # classification (gait recognition)
            logits = gait_head(final_reprs)

            # compute loss
            loss = criterion(logits, labels)  
            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        
        # average training epoch loss
        train_loss = train_loss / (total_samples + 1e-9)
        train_losses.append(train_loss)

        # validation step
        val_loss = 0.0
        val_samples = 0

        # freeze the gait head and cross-attention modules
        gait_head.eval()

        # freeze cross-attention modules as well
        for m in cross_attn_modules_before_T2.values():
            m.eval()
        for m in cross_attn_modules_after_T2.values():
            m.eval()

        with torch.no_grad():
            for batch in val_loader:
                (torso_seq, la_seq, ra_seq, ll_seq, rl_seq, labels) = batch
                torso_seq = torso_seq.to(device)
                la_seq = la_seq.to(device)
                ra_seq = ra_seq.to(device)
                ll_seq = ll_seq.to(device)
                rl_seq = rl_seq.to(device)
                labels = labels.to(device)

                # get T1 encoding for each modality
                encoded_t1 = T1_encoding(t1_map, torso_seq, la_seq, ra_seq, ll_seq, rl_seq)
                encoded_t1_torso = encoded_t1['torso']
                encoded_t1_left_arm = encoded_t1['left_arm']
                encoded_t1_right_arm = encoded_t1['right_arm']
                encoded_t1_left_leg = encoded_t1['left_leg']
                encoded_t1_right_leg = encoded_t1['right_leg']
        
                # get T2 encoding with each other modality O for each modality M
                encoded_T2 = T2_encoding(t2_map, encoded_t1_torso, encoded_t1_left_arm, encoded_t1_right_arm, encoded_t1_left_leg, encoded_t1_right_leg, cross_attn_modules_before_T2)
                encoded_t2_torso = encoded_T2['torso']
                encoded_t2_left_arm = encoded_T2['left_arm']
                encoded_t2_right_arm = encoded_T2['right_arm']
                encoded_t2_left_leg = encoded_T2['left_leg']
                encoded_t2_right_leg = encoded_T2['right_leg']

                # for each modality, do cross-atttention between individual T1 encoding and T2 encoding list
                # here, we do joint cross-attention over the aggregation of 4 other encodings
                # torso
                Q = encoded_t1_torso
                K = V = torch.cat([
                    encoded_t2_torso[0],
                    encoded_t2_torso[1],
                    encoded_t2_torso[2],
                    encoded_t2_torso[3]
                ], dim=1)
                cross_out_torso = cross_attn_modules_after_T2['torso'](Q, K, V)

                # left arm
                Q = encoded_t1_left_arm
                K = V = torch.cat([
                    encoded_t2_left_arm[0],
                    encoded_t2_left_arm[1],
                    encoded_t2_left_arm[2],
                    encoded_t2_left_arm[3]
                ], dim=1)
                cross_out_left_arm = cross_attn_modules_after_T2['left_arm'](Q, K, V)
                # right arm
                Q = encoded_t1_right_arm
                K = V = torch.cat([
                    encoded_t2_right_arm[0],
                    encoded_t2_right_arm[1],
                    encoded_t2_right_arm[2],
                    encoded_t2_right_arm[3]
                ], dim=1)
                cross_out_right_arm = cross_attn_modules_after_T2['right_arm'](Q, K, V)
                # left leg
                Q = encoded_t1_left_leg
                K = V = torch.cat([
                    encoded_t2_left_leg[0],
                    encoded_t2_left_leg[1],
                    encoded_t2_left_leg[2],
                    encoded_t2_left_leg[3]
                ], dim=1)
                cross_out_left_leg = cross_attn_modules_after_T2['left_leg'](Q, K, V)
                # right leg
                Q = encoded_t1_right_leg
                K = V = torch.cat([
                    encoded_t2_right_leg[0],
                    encoded_t2_right_leg[1],
                    encoded_t2_right_leg[2],
                    encoded_t2_right_leg[3]
                ], dim=1)
                cross_out_right_leg = cross_attn_modules_after_T2['right_leg'](Q, K, V)
                # average pooling over time
                pooled_torso = cross_out_torso.mean(dim=1)
                pooled_left_arm = cross_out_left_arm.mean(dim=1)
                pooled_right_arm = cross_out_right_arm.mean(dim=1)
                pooled_left_leg = cross_out_left_leg.mean(dim=1)
                pooled_right_leg = cross_out_right_leg.mean(dim=1)
                # final fused representation
                final_reprs = torch.cat([
                    pooled_torso,
                    pooled_left_arm,
                    pooled_right_arm,
                    pooled_left_leg,
                    pooled_right_leg
                ], dim=-1)
                # classification (gait recognition)
                logits = gait_head(final_reprs)
                # compute loss
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_samples += labels.size(0)
        # average validation epoch loss
        val_loss = val_loss / (val_samples + 1e-9)
        val_losses.append(val_loss)


        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    


    # plotting
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Finetuning - Train and Val Loss')
    plt.savefig('figures/finetuning_train_val_loss.png')


    print("Finetuning complete!")

    # return all necessary model components
    return gait_head, cross_attn_modules_after_T2
