import os

def replace_repo_link(docs_dir, old_link, new_link):
    count = 0
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if old_link in content:
                    new_content = content.replace(old_link, new_link)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated {filepath}")
                    count += 1
    print(f"Total files updated: {count}")

replace_repo_link('docs', 'github.com/SneppX/sneppx-alg', 'github.com/SneppX/sneppx-alg')
