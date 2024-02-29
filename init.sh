# git 설정
git config --global commit.template ./.commit_template
git config --global core.editor "code --wait"
echo -e "\e[34mFin git config\e[0m"

# pre-commit 설정
pip install pre-commit
pre-commit autoupdate
pre-commit install
echo -e "\e[34mFin pre-commit\e[0m"
