# S2-RDSR

Super-resolution of Sentinel-2 2A product using residual dense networks.

To use the trained network, just use

```bash
python RDSR.py S2A_MSIL2A_20221124T101341_N0400_R022_T32TQQ_20221124T151854.zip --x 1000 --y 2000 --size 300
```