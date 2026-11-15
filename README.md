# vram-orbiter
allows running ComfyUI and LM Studio simultaneously by offloading models.  
(позволяет запускать comfyUI и LM-Studio одновременно, выгружая модели)

If a more recent task is running on one of the backends, the other backend will free up its video memory.  
(если в одном из бэкендов запущена более свежая задача, то другой бекенд освободит видеопамять)

run (запуск)
```
python vram_arbiter.py
```
