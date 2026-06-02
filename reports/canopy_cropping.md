# Canopy Cropping

## Summary
|                   |   count |        mean |         std |         min |        25% |         50% |         75% |         max |
|:------------------|--------:|------------:|------------:|------------:|-----------:|------------:|------------:|------------:|
| vegetation_ratio  |     357 |    0.342823 |   0.0809433 |   0.0144875 |   0.356281 |    0.378095 |    0.383176 |    0.396456 |
| canopy_width      |     357 | 1143.63     | 483.194     | 112         | 753        | 1138        | 1473        | 2000        |
| canopy_height     |     357 |  857.098    | 204.296     | 173         | 750        | 1000        | 1000        | 1000        |
| canopy_area_frac  |     357 |    0.519628 |   0.271717  |   0.01708   |   0.301063 |    0.506    |    0.720892 |    1        |
| center_area_frac  |     357 |    0.5      |   0         |   0.5       |   0.5      |    0.5      |    0.5      |    0.5      |
| highveg_area_frac |     357 |    0.5      |   0         |   0.5       |   0.5      |    0.5      |    0.5      |    0.5      |

## Embedding Extraction
| model           | variant   |   dim |   runtime_sec | status   |
|:----------------|:----------|------:|--------------:|:---------|
| dinov2_large    | original  |  1024 |             0 | cached   |
| dinov2_large    | canopy    |  1024 |             0 | cached   |
| dinov2_large    | center    |  1024 |             0 | cached   |
| dinov2_large    | highveg   |  1024 |             0 | cached   |
| siglip_base     | original  |   768 |             0 | cached   |
| siglip_base     | canopy    |   768 |             0 | cached   |
| siglip_base     | center    |   768 |             0 | cached   |
| siglip_base     | highveg   |   768 |             0 | cached   |
| convnextv2_base | original  |  1024 |             0 | cached   |
| convnextv2_base | canopy    |  1024 |             0 | cached   |
| convnextv2_base | center    |  1024 |             0 | cached   |
| convnextv2_base | highveg   |  1024 |             0 | cached   |

## Notes
- ExG masks were used to locate dominant vegetation regions.
- Center/high-vegetation crops are square 1000x1000 crops from each 2000x1000 image.
- Canopy crops are tight vegetation-component boxes with padding; some remain close to full-width when pasture spans the frame.