import os

def create_project_structure():
    structure = {
        "project": {
            "backend": {
                "app": {
                    "models": {"__init__.py": "", "asset.py": ""},
                    "routes": {"__init__.py": "", "portfolio.py": ""},
                    "services": {"__init__.py": "", "optimization.py": ""},
                    "utils": {"__init__.py": "", "data_fetcher.py": ""},
                    "__init__.py": "",
                    "config.py": "",
                },
                "tests": {"test_routes.py": "", "test_services.py": ""},
                "requirements.txt": "",
                "wsgi.py": "",
            },
            "frontend": {
                "public": {"index.html": "<!DOCTYPE html>", "favicon.ico": ""},
                "src": {
                    "components": {"Dashboard.jsx": "", "AssetForm.jsx": ""},
                    "pages": {"HomePage.jsx": "", "ResultsPage.jsx": ""},
                    "services": {"api.js": ""},
                    "App.jsx": "",
                    "index.js": "",
                    "styles.css": "",
                },
                "package.json": "",
                "vite.config.js": "",
            },
            "docker-compose.yml": "",
            "README.md": "",
            ".gitignore": "",
        }
    }

    def create_files(base_path, content):
        for name, value in content.items():
            path = os.path.join(base_path, name)
            if isinstance(value, dict):  # Create folder
                os.makedirs(path, exist_ok=True)
                create_files(path, value)
            else:  # Create file
                with open(path, "w") as f:
                    f.write(value)

    base_dir = os.getcwd()  # Current directory
    create_files(base_dir, structure)
    print(f"Project structure created at {base_dir}/project")

# Uruchom funkcję
if __name__ == "__main__":
    create_project_structure()
