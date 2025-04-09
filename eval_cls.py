import numpy as np
import argparse
import torch
from models import cls_model
from utils import create_dir, viz_cls
import os
import pytorch3d.transforms

# Define class names
class_names = {
    0: "chair",
    1: "vase",
    2: "lamp",
    3: "guitar",
    4: "laptop",
    5: "motorbike"
}

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument("--eval_all", type=bool, default=False, help="Evaluate all objects")
    parser.add_argument("--input_all", type=bool, default=False, help="Input all objects")
    parser.add_argument("--eval_num", type=int, default=10, help="Evaluate how many objects")
    
    # Experiment parameters
    parser.add_argument('--angle', type=float, default=0, help='Rotation angle in degrees for robustness test')
    parser.add_argument('--point_experiment', type=bool, default=False, help='Run point number experiment')
    parser.add_argument('--min_points', type=int, default=1000, help='Minimum number of points for point experiment')
    parser.add_argument('--max_points', type=int, default=10000, help='Maximum number of points for point experiment')
    parser.add_argument('--point_steps', type=int, default=5, help='Number of steps for point experiment')

    return parser

def evaluate_model(model, test_data, test_label, args):
    """Evaluate model on test data with optional rotation."""
    # Apply rotation if specified
    if args.angle != 0:
        print(f"\nEvaluating with {args.angle} degree rotation...")
        # Convert angle to radians
        angle_rad = torch.tensor(args.angle * np.pi / 180.0)
        # Create rotation matrix around Y axis
        rot = pytorch3d.transforms.euler_angles_to_matrix(
            torch.tensor([0.0, angle_rad, 0.0]), "XYZ"
        ).to(args.device)
        # Apply rotation to all points
        test_data = torch.matmul(test_data, rot)

    # Make predictions
    with torch.no_grad():
        pred_label = model(test_data)
        pred_label = torch.argmax(pred_label, dim=1)

    # Compute accuracy
    accuracy = pred_label.eq(test_label.data).cpu().sum().item() / test_label.size()[0]
    return accuracy, pred_label

def run_point_experiment(model, test_data, test_label, args):
    """Run experiment with different numbers of points."""
    print("\nRunning point number experiment...")
    point_counts = np.linspace(args.min_points, args.max_points, args.point_steps, dtype=int)
    accuracies = []

    for num_points in point_counts:
        print(f"\nEvaluating with {num_points} points...")
        # Sample points
        ind = np.random.choice(10000, num_points, replace=False)
        sampled_data = test_data[:, ind, :]
        
        # Evaluate
        accuracy, _ = evaluate_model(model, sampled_data, test_label, args)
        accuracies.append(accuracy)
        print(f"Accuracy with {num_points} points: {accuracy:.4f}")

    # Plot results
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(point_counts, accuracies, 'b-o')
    # plt.xlabel('Number of Points')
    # plt.ylabel('Accuracy')
    # plt.title('Model Accuracy vs Number of Points')
    # plt.grid(True)
    # plt.savefig(f"{args.output_dir}/point_experiment.png")
    # plt.close()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # Initialize Model
    model = cls_model(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Load test data
    test_data = torch.from_numpy(np.load(args.test_data)).float().to(args.device)
    test_label = torch.from_numpy(np.load(args.test_label)).long().to(args.device)

    # Run point experiment if requested
    if args.point_experiment:
        run_point_experiment(model, test_data, test_label, args)
    else:
        # Sample points for normal evaluation
        ind = np.random.choice(10000, args.num_points, replace=False)
        test_data = test_data[:, ind, :]

        # Evaluate model
        accuracy, pred_label = evaluate_model(model, test_data, test_label, args)
        print(f"\nTest accuracy: {accuracy:.4f}")

        # Check all objects and visualize mismatches
        if args.eval_all:
            print("\nChecking all objects:")
            print("idx\tGT\t\tPred\t\tMatch")
            print("-" * 40)
            num = 0
            if args.input_all: 
                num = len(test_label)
            else:
                num = args.eval_num
            for i in range(num):
                gt = test_label[i].item()
                pred = pred_label[i].item()
                match = "✓" if gt == pred else "✗"
                print(f"{i}\t{class_names[gt]}\t{class_names[pred]}\t{match}")
                
                # Visualize if prediction doesn't match ground truth
                # if gt != pred:
                #     print(f"\nVisualizing mismatch for object {i}:")
                #     viz_cls(args, test_data[i], gt, f"{args.output_dir}/mismatch_{args.exp_name}_{i}_gt_{class_names[gt]}.gif", args.device)
                #     viz_cls(args, test_data[i], pred, f"{args.output_dir}/mismatch_{args.exp_name}_{i}_pred_{class_names[pred]}.gif", args.device)

        # Visualize specified object
        if not args.eval_all:
            viz_cls(args, test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
            viz_cls(args, test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)

