#!/bin/bash
# UV execution script for training all forecasting models
# This script uses UV to run the training pipeline with proper dependency management

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Forecasting Pipeline - UV Execution${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: UV is not installed.${NC}"
    echo "Please install UV: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${YELLOW}Using UV version:${NC}"
uv --version
echo ""

# Check if data files exist
TRAIN_DATA="data/merged_filter_ingestion_2024.csv"
TEST_DATA="data/merged_filter_ingestion_2025.csv"

if [ ! -f "$TRAIN_DATA" ]; then
    echo -e "${RED}Error: Training data file not found: $TRAIN_DATA${NC}"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo -e "${RED}Error: Test data file not found: $TEST_DATA${NC}"
    exit 1
fi

echo -e "${GREEN}Data files found:${NC}"
echo "  - Training: $TRAIN_DATA"
echo "  - Test: $TEST_DATA"
echo ""

# Run the training script using UV
echo -e "${YELLOW}Starting training pipeline...${NC}"
echo -e "${YELLOW}Forecast target: FY 2025-26 (April 2025 to March 2026)${NC}"
echo ""

# Use UV to run the training script
uv run python training/train_all_models.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results are saved in: training/results/"
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Training failed with errors!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

