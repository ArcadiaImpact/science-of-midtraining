import os, sys
from huggingface_hub import HfApi
src, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=src, repo_id=repo,
                  allow_patterns=["adapter_*", "*.json", "*.jinja", "tokenizer*", "*.md"])
print("PUBLISHED", repo)
