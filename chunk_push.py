import os
import subprocess
import shutil

# Configuration
CHUNK_SIZE = 10000
REPO_URL = "https://github.com/Djyoti0210/AffectNet-.git"
AUTHOR_NAME = "Dhamma Djyoti dhawase"
AUTHOR_EMAIL = "dhammadjyoti@example.com"
BRANCH = "main"

def run_cmd(cmd, **kwargs):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, check=True, shell=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise

def setup_git():
    if os.path.exists(".git"):
        print("Removing existing .git directory... this might take a minute.")
        shutil.rmtree(".git", ignore_errors=True)
    
    run_cmd("git init")
    run_cmd(f'git config user.name "{AUTHOR_NAME}"')
    run_cmd(f'git config user.email "{AUTHOR_EMAIL}"')
    run_cmd(f"git remote add origin {REPO_URL}")
    run_cmd("git config http.postBuffer 524288000")
    run_cmd("git config http.version HTTP/1.1")
    run_cmd(f"git branch -M {BRANCH}")

def get_all_files():
    all_files = []
    for root, _, files in os.walk("."):
        if ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            # Format path properly for git
            filepath = os.path.join(root, f).replace("\\", "/")
            if filepath.startswith("./"):
                filepath = filepath[2:]
            all_files.append(filepath)
    return all_files

def main():
    setup_git()
    all_files = get_all_files()
    
    # First, add everything except the human_annotated folder to get the base structure pushed
    print("Pushing base structure...")
    base_files = [f for f in all_files if not f.startswith("human_annotated")]
    human_files = [f for f in all_files if f.startswith("human_annotated")]
    
    with open("chunk.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(base_files))
    
    if base_files:
        run_cmd("git add --pathspec-from-file=chunk.txt")
        run_cmd('git commit -m "Initial commit: Base project structure"')
        run_cmd(f"git push -u origin {BRANCH} -f")
    
    # Now chunk the huge folder
    print(f"Total files to chunk: {len(human_files)}")
    
    chunks = [human_files[i:i + CHUNK_SIZE] for i in range(0, len(human_files), CHUNK_SIZE)]
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} files)...")
        with open("chunk.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(chunk))
        
        run_cmd("git add --pathspec-from-file=chunk.txt")
        run_cmd(f'git commit -m "Upload dataset chunk {i+1}/{len(chunks)}"')
        
        # We don't force push here, just push the new commit
        print(f"Pushing chunk {i+1}...")
        # Add retries for push in case of temporary network failure
        max_retries = 3
        for attempt in range(max_retries):
            try:
                run_cmd(f"git push origin {BRANCH}")
                break
            except Exception as e:
                print(f"Push failed on attempt {attempt+1}. Retrying...")
                if attempt == max_retries - 1:
                    raise
    
    print("ALL FILES SUCCESSFULLY PUSHED!")
    
if __name__ == "__main__":
    main()
