set -e

# Create the output directories.
CURRENT_DIR=$(pwd)
OUTPUT_DIR="${CURRENT_DIR}/data/mscoco"
RAW_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${RAW_DIR}"

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=$1
  local FILENAME=$2

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to ${RAW_DIR}"
    wget -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  unzip -nq "${FILENAME}"
}

cd ${RAW_DIR}

# Download the images.
BASE_IMAGE_URL="http://msvocds.blob.core.windows.net/coco2014"

TRAIN_IMAGE_FILE="train2014.zip"
#download_and_unzip ${BASE_IMAGE_URL} ${TRAIN_IMAGE_FILE}
TRAIN_IMAGE_DIR="${RAW_DIR}/train2014"

VAL_IMAGE_FILE="val2014.zip"
#download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE}
VAL_IMAGE_DIR="${RAW_DIR}/val2014"

# Download the captions.
BASE_CAPTIONS_URL="http://msvocds.blob.core.windows.net/annotations-1-0-3"
CAPTIONS_FILE="captions_train-val2014.zip"
#download_and_unzip ${BASE_CAPTIONS_URL} ${CAPTIONS_FILE}
TRAIN_CAPTIONS_FILE="${RAW_DIR}/annotations/captions_train2014.json"
VAL_CAPTIONS_FILE="${RAW_DIR}/annotations/captions_val2014.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="./tutorials/img2txt/ops/toTFRecords.py"
python3 ${BUILD_SCRIPT}\
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --train_captions_file="${TRAIN_CAPTIONS_FILE}" \
  --val_captions_file="${VAL_CAPTIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}/tfrecords" \
  --word_counts_output_file="${OUTPUT_DIR}/words.txt" \