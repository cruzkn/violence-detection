#!/bin/bash
# =============================================================================
# github_push.sh
# Initialises git repo and pushes to github.com/cruzkn/violence-detection
# Run once after setup: chmod +x github_push.sh && ./github_push.sh
# =============================================================================

set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

GITHUB_USER="cruzkn"
REPO_NAME="violence-detection"
REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Push Violence Detection Project to GitHub              ║"
echo "║   → github.com/${GITHUB_USER}/${REPO_NAME}              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check git is installed ───────────────────────────────────────────────────
if ! command -v git &>/dev/null; then
    echo -e "${YELLOW}Git not found. Installing via Homebrew...${NC}"
    brew install git
fi

# ── Initialise git if needed ─────────────────────────────────────────────────
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initialising git repository...${NC}"
    git init
    git branch -M main
    echo -e "${GREEN}✓ Git initialised${NC}"
else
    echo -e "${GREEN}✓ Git already initialised${NC}"
fi

# ── Set remote ───────────────────────────────────────────────────────────────
if git remote get-url origin &>/dev/null; then
    echo -e "${GREEN}✓ Remote 'origin' already set${NC}"
else
    echo -e "${YELLOW}Adding remote origin → ${REMOTE_URL}${NC}"
    git remote add origin "${REMOTE_URL}"
fi

# ── Stage all files ───────────────────────────────────────────────────────────
echo -e "${YELLOW}Staging files...${NC}"
git add .

# ── Show what will be committed ──────────────────────────────────────────────
echo -e "\n${CYAN}Files to commit:${NC}"
git diff --cached --name-only | sed 's/^/  /'

# ── Commit ───────────────────────────────────────────────────────────────────
echo ""
read -p "Enter commit message (or press Enter for default): " MSG
MSG="${MSG:-Add violence detection thesis implementation}"

git commit -m "${MSG}"

# ── Push ─────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Pushing to GitHub...${NC}"
echo -e "${CYAN}NOTE: GitHub will ask for your username + personal access token${NC}"
echo -e "${CYAN}Get token at: https://github.com/settings/tokens → Generate new token (classic)${NC}"
echo -e "${CYAN}Required scopes: repo, workflow${NC}\n"

git push -u origin main

echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════╗"
echo "║   ✓ Successfully pushed to GitHub!                      ║"
echo "║   → https://github.com/${GITHUB_USER}/${REPO_NAME}     ║"
echo "╚══════════════════════════════════════════════════════════╝${NC}"
