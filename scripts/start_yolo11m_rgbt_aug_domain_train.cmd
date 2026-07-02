@echo off
cd /d D:\cv
set MPLBACKEND=Agg
set PYTHONUNBUFFERED=1
echo [%date% %time%] starting yolo11m rgbt train > work_dirs\yolo11m_rgbt_aug_domain_w0_launcher.log
"D:\Anaconda\envs\yolo_gaiic\python.exe" -u scripts\train_rgbt_yolo.py ^
  --model yolo11m.pt ^
  --data configs\gaiic_yolo_rgbt_aug_domain.yaml ^
  --imgsz 640 ^
  --epochs 200 ^
  --batch 8 ^
  --device 0 ^
  --workers 0 ^
  --project D:/cv/work_dirs ^
  --name yolo11m_rgbt_aug_domain_w0 ^
  --lr0 0.01 ^
  --lrf 0.01 ^
  --close-mosaic 10 ^
  --patience 100 ^
  --seed 0 ^
  --exist-ok ^
  1> work_dirs\yolo11m_rgbt_aug_domain_w0_train.out.log ^
  2> work_dirs\yolo11m_rgbt_aug_domain_w0_train.err.log
echo [%date% %time%] python exit code %errorlevel% >> work_dirs\yolo11m_rgbt_aug_domain_w0_launcher.log
