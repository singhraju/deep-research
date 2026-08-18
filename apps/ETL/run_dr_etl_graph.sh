#!/bin/sh

# DR ETL Pipeline (LangGraph) Runner Script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
ENV=""
USE_CSV=""
CSV_FILE="SampleData-dev-deep-research.csv"
LOB="gbd"
STATSCL_MDL_CD="IP AUTH"
SNAP_YEAR_MNTH_NBR=""
TRND_TM_PRD_CD=""
LOB_SHRT_DESC=""

usage() {
    echo "Usage: $0 <environment> [options]"
    echo ""
    echo "Environments: dv, ts, pl, pr"
    echo ""
    echo "Options:"
    echo "  --use-csv"
    echo "  --csv-file PATH"
    echo "  --lob LOB (gbd | nogbd)"
    echo "  --statscl-mdl-cd MODEL   Statistical model code (default: IP AUTH)"    
    echo "  --snap-year-mnth-nbr SNAP_YEAR_MNTH_NBR SNAP YEAR MONTH Number (default: None)"
    echo "  --trnd-tm-prd-cd TRND_TM_PRD_CD Trend period code (optional, e.g., R3, R6, R12)"
    echo "  --lob-shrt-desc LOB_SHRT_DESC LOB short description (optional, e.g., Commercial_Individual)"
    exit 1
}

# Check env
if [ -z "$1" ]; then
    usage
fi

ENV=$1
shift

# Validate ENV (no regex in sh)
if [ "$ENV" != "dv" ] && [ "$ENV" != "ts" ] && [ "$ENV" != "pl" ] && [ "$ENV" != "pr" ]; then
    echo "${RED}Error: Invalid environment '$ENV'${NC}"
    usage
fi

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --use-csv)
            USE_CSV="--use-csv"
            shift
            ;;
        --csv-file)
            CSV_FILE=$2
            shift 2
            ;;
        --lob)
            LOB=$2
            shift 2
            ;;
        --statscl-mdl-cd)
            STATSCL_MDL_CD=$2
            shift 2
            ;;            
        --snap-year-mnth-nbr)
            SNAP_YEAR_MNTH_NBR=$2
            shift 2
            ;;
        --trnd-tm-prd-cd)
            TRND_TM_PRD_CD=$2
            shift 2
            ;;
        --lob-shrt-desc)
            LOB_SHRT_DESC=$2
            shift 2
            ;;                    
        *)
            echo "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate LOB
if [ "$LOB" != "gbd" ] && [ "$LOB" != "nogbd" ]; then
    echo "${RED}Error: Invalid LOB '$LOB'${NC}"
    exit 1
fi

# Print config
echo "${CYAN}========================================${NC}"
echo "${CYAN}DR ETL Pipeline (LangGraph)${NC}"
echo "${CYAN}========================================${NC}"
echo "${GREEN}Environment: $ENV${NC}"
echo "${GREEN}LOB: $LOB${NC}"
echo "${GREEN}Statistical Model: $STATSCL_MDL_CD${NC}"
if [ -n "$SNAP_YEAR_MNTH_NBR" ]; then
    echo "${GREEN}SNAP_YEAR_MNTH_NBR: $SNAP_YEAR_MNTH_NBR${NC}"
fi
if [ -n "$TRND_TM_PRD_CD" ]; then
    echo "${GREEN}TRND_TM_PRD_CD: $TRND_TM_PRD_CD${NC}"
fi
if [ -n "$LOB_SHRT_DESC" ]; then
    echo "${GREEN}LOB_SHRT_DESC: $LOB_SHRT_DESC${NC}"
fi

if [ -n "$USE_CSV" ]; then
    echo "${YELLOW}Mode: CSV File${NC}"
    echo "${YELLOW}CSV: $CSV_FILE${NC}"
else
    echo "${GREEN}Mode: Snowflake${NC}"
fi

echo "${CYAN}========================================${NC}"
echo ""

# Activate conda (POSIX-safe)
echo "${YELLOW}Activating deep research venv...${NC}"

# Activate to venv
if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
    echo "${GREEN}Activated .venv${NC}"
elif [ -f "venv/bin/activate" ]; then
    . venv/bin/activate
    echo "${GREEN}Activated venv${NC}"
elif [ -f "../../.venv/bin/activate" ]; then
    . ../../.venv/bin/activate
    echo "${GREEN}Activated ../../.venv${NC}"
else
    echo "${YELLOW}No virtual environment found, using system Python${NC}"
fi

# Run pipeline
echo "${CYAN}Starting pipeline...${NC}"

SNAP_ARG=""
if [ -n "$SNAP_YEAR_MNTH_NBR" ]; then
    SNAP_ARG="--snap-year-mnth-nbr '$SNAP_YEAR_MNTH_NBR'"
fi

TRND_ARG=""
if [ -n "$TRND_TM_PRD_CD" ]; then
    TRND_ARG="--trnd-tm-prd-cd '$TRND_TM_PRD_CD'"
fi

LOB_DESC_ARG=""
if [ -n "$LOB_SHRT_DESC" ]; then
    LOB_DESC_ARG="--lob-shrt-desc '$LOB_SHRT_DESC'"
fi

# Use eval to properly handle quoted arguments
eval python /app/apps/ETL/dr_etl_graph.py --env "'$ENV'" --lob "'$LOB'" --statscl-mdl-cd "'$STATSCL_MDL_CD'" $USE_CSV $SNAP_ARG $TRND_ARG $LOB_DESC_ARG --csv-file "'$CSV_FILE'" 
EXIT_CODE=$?

# Status
if [ $EXIT_CODE -eq 0 ]; then
    echo "${GREEN}Pipeline completed successfully${NC}"
else
    echo "${RED}Pipeline failed with code $EXIT_CODE${NC}"
fi

echo "${YELLOW}Deactivating env...${NC}"
