FILE=$1

if [[ $FILE != "cityscapes" && $FILE != "facades" ]]; then
  echo "Available datasets are cityscapes, facades"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR
echo "Downloading $URL dataset..." to $TARGET_DIR
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE

mkdir -p ./train_list ./val_list
find "${TARGET_DIR}train" -type f -name "*.jpg" |sort -V > ./train_list/${FILE}_train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" |sort -V > ./val_list/${FILE}_val_list.txt
