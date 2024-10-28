import subprocess
import ffmpeg
import os
import shutil


def split_video(input_video: str, workspace_path: str, duration: int = 10):    
    workspace_path = workspace_path.replace("\\", "/")
    workspace_image_path = f"{workspace_path}/output_frames_folder"
    if not os.path.isfile(input_video):
        raise FileNotFoundError("Missing input video")

    # duration = get_video_length(input_video)

    print(f"Running FFMPEG with {20//duration} fps") 

    # Define the command and arguments
    command = [
        "wsl",
        "-e", 
        "ffmpeg",
        "-i",
        input_video,
        "-vf", 
        f"fps={20//duration}",
        f"{workspace_image_path}/frame_%04d.png"
    ]
    
    # Create frames folder if it doesn't exist
    if not os.path.isdir(workspace_image_path):
        os.makedirs(workspace_image_path)

    try:
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

    except Exception as e:
        print("An error occurred:", e)

def setup_workspace(file_path: str):
    # Extract the filename without the extension and directory path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    workspace_dir = os.path.join("workspaces", f"{base_name}_workspace")

    # Check if the workspace directory exists
    if os.path.isdir(workspace_dir):
        # Clear the directory if it exists
        print(f"Clearing existing workspace: {workspace_dir}")    
        for filename in os.listdir(workspace_dir):            
            file_path = os.path.join(workspace_dir, filename)            
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
    else:
        # Create the directory if it doesn't exist
        print(f"Creating new workspace: {workspace_dir}")
        os.makedirs(workspace_dir)

    return workspace_dir