
for file in $(ls /media/josh/nebula_josh/hdr/fall2021_hdr_mp4/*.mp4);
do
    echo "$file"
    res=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 -i "$file")
    fps=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate -i "$file") 
    framenos=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "$file")
    printf '%s\n' "$file" "$res" "$fps" "$framenos" | paste -sd ',' >> "fall2021_hdr_res_fps_fnos_nebula.csv"
done

