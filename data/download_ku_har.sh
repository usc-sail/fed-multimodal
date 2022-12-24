source ../system.cfg
echo "Data folder: "$data_dir

if [[ ! -e $data_dir ]]; then
    mkdir $data_dir
fi

cd $data_dir && mkdir ku-har
cd ku-har

wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/45f952y38r-5.zip

unzip 45f952y38r-5.zip

if [[ ! -e "Trimmed_interpolated_data" ]]; then
    mkdir "Trimmed_interpolated_data"
fi
cp 2.Trimmed_interpolated_data.zip Trimmed_interpolated_data/

cd Trimmed_interpolated_data
unzip 2.Trimmed_interpolated_data.zip
cd .. && rm 45f952y38r-5.zip