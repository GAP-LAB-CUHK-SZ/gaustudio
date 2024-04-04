colmap feature_extractor \
    --database_path=${1}/database.db \
    --image_path=${1}/images \
    --ImageReader.camera_model=SIMPLE_RADIAL \
    --ImageReader.single_camera=true \
    --SiftExtraction.use_gpu=true \
    --SiftExtraction.num_threads=32

colmap sequential_matcher \
    --database_path=${1}/database.db \
    --SiftMatching.use_gpu=true

mkdir -p ${1}/sparse
colmap mapper \
    --database_path=${1}/database.db \
    --image_path=${1}/images \
    --output_path=${1}/sparse

cp ${1}/sparse/0/*.bin ${1}/sparse/
for path in ${1}/sparse/*/; do
    m=$(basename ${path})
    if [ ${m} != "0" ]; then
        colmap model_merger \
            --input_path1=${1}/sparse \
            --input_path2=${1}/sparse/${m} \
            --output_path=${1}/sparse
        colmap bundle_adjuster \
            --input_path=${1}/sparse \
            --output_path=${1}/sparse
    fi
done

colmap image_undistorter \
    --image_path=${1}/images \
    --input_path=${1}/sparse \
    --output_path=${1} \
    --output_type=COLMAP