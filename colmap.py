import subprocess
import os
from preprocessing import split_video, setup_workspace
from postprocessing import ply_to_glb

# Change this path to point to the bat file for your COLMAP installation (may be different for mac)
COLMAP_PATH = "C:\\Users\\natha\\OneDrive\\Desktop\\COLMAP.bat"

def generate_ply(video_name: str, duration: int):
    workspace_path = setup_workspace(video_name)
    workspace_image_path = f"{workspace_path}/output_frames_folder"
    # Define the command and arguments
    command = [
        COLMAP_PATH,
        "automatic_reconstructor",
        "--use_gpu",
        "0",
        "--single_camera",
        "1",
        "--quality",
        "low",
        "--workspace_path",
        workspace_path,
        "--image_path",
        workspace_image_path
    ]

    try:
        # Split the video
        split_video(video_name, workspace_path, duration=duration)

        # Check some conditiosn
        if not os.path.isdir(workspace_path) or not os.path.isdir(workspace_image_path):
            raise FileNotFoundError("Missing workspaces")
            
        # Run the command and print output in real-time
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Continuously read and print output
        for line in process.stdout:
            print(line, end='')  # Print each line as it comes in

        # Wait for the process to complete
        process.wait()

        # Check if the process finished successfully
        if process.returncode == 0:
            print("Command executed successfully.")
        else:
            print("Command failed with return code:", process.returncode)
        
        normal_path = workspace_path.replace("\\", "/")
        return f"{normal_path}/dense/0/meshed-poisson.ply"

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    video_name = "mouse.MOV"
    duration = 10

    ply_file = generate_ply(video_name, duration)
    # ply_file = "workspaces/tissue_workspace/dense/0/meshed-poisson.ply"
        
    ply_to_glb(ply_file=ply_file, glb_file="output.glb")