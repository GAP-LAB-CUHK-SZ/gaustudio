video_path=${1}
image_path=${2}
downsample_rate=${3}
mkdir -p ${image_path}
ffmpeg -i ${video_path} -vf "select=not(mod(n\,${downsample_rate}))" -vsync vfr -q:v 2 ${image_path}/%06d.jpg