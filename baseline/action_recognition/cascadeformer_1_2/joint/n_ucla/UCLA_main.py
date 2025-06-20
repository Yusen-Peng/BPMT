import torch
import argparse
from base_dataset import ActionRecognitionDataset
from pretraining import train_T1, BaseT1
from UCLA_finetuning import load_T1, finetuning, GaitRecognitionHead, evaluate, load_T2, load_cross_attn
#from first_phase_baseline import BaseT1, train_T1
#from second_phase_baseline import BaseT2, train_T2, load_T1
#from finetuning import GaitRecognitionHead, finetuning, load_T2, load_cross_attn
from UCLA_utils import set_seed, NUM_JOINTS_NUCLA, split_train_val
from SF_UCLA_loader import SF_UCLA_Dataset, skateformer_collate_fn

def count_all_parameters(T1, T2, cross_attn, gait_head) -> int:
    """
    Counts the total number of parameters in the T1, T2, cross-attention, and gait head models.
    
    Args:
        T1: T1 transformer model
        T2: T2 transformer model
        cross_attn: CrossAttention module
        gait_head: GaitRecognitionHead module

    Returns:
        total_params: Total number of parameters across all models
    """
    total_params = sum(p.numel() for p in T1.parameters())
    total_params += sum(p.numel() for p in T2.parameters())
    total_params += sum(p.numel() for p in cross_attn.parameters())
    total_params += sum(p.numel() for p in gait_head.parameters())
    return total_params


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Training")
    parser.add_argument("--pretrain", action='store_true', help="Run the stage of pretraining")
    parser.add_argument("--root_dir", type=str, default="N_UCLA/", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--class_specific_split", action='store_true', help="Use class-specific split for training and validation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()

def main():
    set_seed(42)

    # are we actually training or just evaluating?
    TRAIN = False

    # masking_strategy = "frame", "global_joint"
    masking_strategy = "global_joint"
    mask_ratio = 0.3
    val_ratio = 0.05
    args = parse_args()
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    device = args.device
    pretrain = args.pretrain

    print(f"pretrain?: {pretrain}")

    # transformer parameters
    hidden_size = 256
    n_heads = 8
    num_layers = 4
    print(f"hidden_size: {hidden_size}")
    print(f"n_heads: {n_heads}")
    print(f"num_layers: {num_layers}")
    print(f"batch_size: {batch_size}")

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting NW-UCLA dataset processing on {device}...")
    print("=" * 50)
    
    train_data_path = 'N-UCLA_processed/'
    train_label_path = 'N-UCLA_processed/train_label.pkl'

    data_type = 'j'
    repeat = 25     # 10, 15
    p = 0.3 # 0.1, 0.5

    print(f"[INFO]: proability of dropping regularization: {p}")
    print(f"[INFO]: data being repeated by {repeat} times")

    train_dataset_pre = SF_UCLA_Dataset(
        data_path=train_data_path,
        label_path=train_label_path,
        data_type=data_type,
        window_size=-1, 
        partition=True, 
        repeat=repeat,
        p=p,
        debug=False
    )

    train_seq = []
    train_lbl = []

    for i in range(len(train_dataset_pre)):
        data, _, label, _ = train_dataset_pre[i]
        data_tensor = torch.from_numpy(data)
        train_seq.append(data_tensor)
        train_lbl.append(label)

    print(f"Collected {len(train_seq)} sequences for train + val.")
    print(f"Each sequence shape: {train_seq[0].shape}")  # (64, 60)

    train_seq, train_lbl, _, _ = split_train_val(train_seq, train_lbl, val_ratio=val_ratio)
    test_data_path = 'N-UCLA_processed/'
    test_label_path = 'N-UCLA_processed/val_label.pkl'

    test_dataset_pre = SF_UCLA_Dataset(
        data_path=test_data_path,
        label_path=test_label_path,
        data_type='j', 
        window_size=64, 
        partition=True, 
        repeat=1, 
        p=0.0, 
        debug=False
    )

    test_seq = []
    test_lbl = []
    for i in range(len(test_dataset_pre)):
        data, _, label, _ = test_dataset_pre[i]

        data_tensor = torch.from_numpy(data)

        test_seq.append(data_tensor)
        test_lbl.append(label)
    
    num_classes = max(train_lbl + test_lbl) + 1
    train_dataset = ActionRecognitionDataset(train_seq, train_lbl)
    val_dataset = ActionRecognitionDataset(test_seq, test_lbl)

    train_finetuning_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=skateformer_collate_fn
    )

    val_finetuning_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=skateformer_collate_fn
    )

    if TRAIN: 

        if pretrain:
            """
                pretraining on the whole dataset
            """

            print("\n==========================")
            print("Starting Pretraining...")
            print("==========================")
            
            # instantiate the model
            three_d = True
            model = BaseT1(
                num_joints=NUM_JOINTS_NUCLA,
                three_d=three_d,
                d_model=hidden_size,
                nhead=n_heads,
                num_layers=num_layers,
            ).to(device)
            
            # training
            # dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
            print(f"[INFO] Mask ratio: {mask_ratio * 100}%")
            print(f"[INFO] train/val split ratio: {val_ratio * 100}%")
            lr = 1e-4
            train_T1(
                masking_strategy=masking_strategy,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                model=model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                mask_ratio=mask_ratio,
                device=device
            )

            print("[TEST] testing global joint masking" + "=" * 40)
            # save pretrained model
            torch.save(model.state_dict(), "action_checkpoints/NUCLA_pretrained.pt")

            print("Aha! pretraining is done!")
            print("=" * 100)
        
        
        print("=" * 100)
        print("=" * 100)
        print("=" * 100)


        # load T1 models
        three_d = True
        t1 = load_T1(
            model_path="action_checkpoints/NUCLA_pretrained.pt",
            num_joints=NUM_JOINTS_NUCLA,
            three_d=three_d,
            d_model=hidden_size,
            nhead=n_heads,
            num_layers=num_layers,
            freeze=True,
            device=device
        )

        print("pretrained model loaded successfully!")

        gait_head_template = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes).to(device)

        freezeT1 = False
        unfreeze_layers = None # freeze all layers

        if freezeT1 and (unfreeze_layers is None):
            print("[INFO] freezing the entire T1 model...")
        elif freezeT1 and (unfreeze_layers is not None):
            print("[INFO] layerwise finetuning...")
            print(f"[INFO] unfreezing layers: {unfreeze_layers}...")
        elif not freezeT1:
            print("[INFO] finetuning the entire T1 model...")

        # finetuning learning rate
        fn_lr = 3e-5 # 3e-5
        wd = 1e-2 # 1e-2
        trained_T1, trained_T2, train_cross_attn, train_head = finetuning(
            train_loader=train_finetuning_dataloader,
            val_loader=val_finetuning_dataloader,
            t1=t1,
            gait_head=gait_head_template,
            d_model=hidden_size,
            nhead=n_heads,
            num_layers=num_layers,
            num_epochs=num_epochs,
            lr=fn_lr,
            wd=wd,
            freezeT1=freezeT1,
            unfreeze_layers=unfreeze_layers,
            device=device
        )

        print("Aha! Finetuning completed successfully!")
        if unfreeze_layers is not None:
            print(f"[INFO] Unfreezing layers: {unfreeze_layers}...")
        
        final_acc = evaluate(
            data_loader=val_finetuning_dataloader,
            t1=trained_T1,
            t2=trained_T2,
            cross_attn=train_cross_attn,
            gait_head=train_head,
            device=device
        )
        print("💎"* 20)
        print(f"[INFO] Final accuracy after finetuning: {final_acc:.4f}")

        # save the finetuned models
        torch.save(trained_T2.state_dict(), "action_checkpoints/NUCLA_finetuned_T2.pt")
        torch.save(train_cross_attn.state_dict(), "action_checkpoints/NUCLA_finetuned_cross_attn.pt")
        torch.save(train_head.state_dict(), "action_checkpoints/NUCLA_finetuned_head.pt")
        torch.save(trained_T1.state_dict(), "action_checkpoints/NUCLA_finetuned_T1.pt")

        print("Aha! finetuned models saved successfully!")

    # load T1 model
    unfreeze_layers = "entire"
    if unfreeze_layers is None:
        print("************Freezing all layers")
        t1 = load_T1("action_checkpoints/NUCLA_pretrained.pt", 
                    num_joints=NUM_JOINTS_NUCLA,
                    three_d=True,
                    d_model=hidden_size, 
                    nhead=n_heads, 
                    num_layers=num_layers, 
                    device=device,
                    freeze=True
                )
    else:
        t1 = load_T1("action_checkpoints/NUCLA_finetuned_T1.pt",
                    num_joints=NUM_JOINTS_NUCLA,
                    three_d=True,
                    d_model=hidden_size, 
                    nhead=n_heads, 
                    num_layers=num_layers, 
                    device=device, 
                )
        print(f"************Unfreezing layers: {unfreeze_layers}")
    

    
    t2 = load_T2("action_checkpoints/NUCLA_finetuned_T2.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, device=device)
    # load the cross attention module
    cross_attn = load_cross_attn("action_checkpoints/NUCLA_finetuned_cross_attn.pt", d_model=hidden_size, device=device)

    # load the gait recognition head
    gait_head = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes)
    gait_head.load_state_dict(torch.load("action_checkpoints/NUCLA_finetuned_head.pt", map_location=device))
    gait_head = gait_head.to(device)


    print("Aha! All models loaded successfully!")
    print("=" * 100)
    print("the total number of parameters in the model is: ")
    total_params = count_all_parameters(t1, t2, cross_attn, gait_head)
    print(f"Total parameters: {total_params:,}")

    # evaluate the model
    accuracy = evaluate(
        val_finetuning_dataloader,
        t1,
        t2,
        cross_attn,
        gait_head,
        device=device
    )

    print("[INFO] Evaluation completed!")
    print(f"Final Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()