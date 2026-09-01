---
license: cc-by-4.0
language:
- pl
pretty_name: "JuDDGES-pl: Polish Court Judgment Corpus for Civil-Law Legal NLP"
tags:
- legal
- polish
- civil-law
- silver-labels
- llm-extracted
- judgments
- JuDDGES
size_categories:
- 1M<n<10M
task_categories:
- text-generation
- text-classification
- summarization
- feature-extraction
configs:
- config_name: pl-court
  data_files:
  - split: train
    path:
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0002.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0003.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0004.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0005.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0006.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0007.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0008.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0009.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0010.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0011.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0012.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0013.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0014.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0015.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0016.parquet"
- config_name: pl-nsa
  data_files:
  - split: train
    path:
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0002.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0003.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0004.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0005.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0006.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0007.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0008.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0009.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0010.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0011.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0012.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0013.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0014.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0015.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0016.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0017.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0018.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0019.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0020.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0021.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0022.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0023.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0024.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0025.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0026.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0027.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0028.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0029.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0030.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0031.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0032.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0033.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0034.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0035.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0036.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0037.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0038.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0039.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0040.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0041.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0042.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0043.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0044.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0045.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0046.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0047.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0048.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0049.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0050.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0051.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0052.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0053.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0054.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0055.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0056.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0057.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0058.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0059.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0060.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0061.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0062.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0063.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0064.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0065.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0066.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0067.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0068.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0069.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0070.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0071.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0072.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0073.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0074.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0075.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0076.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0077.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0078.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0079.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0080.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0081.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0082.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0083.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0084.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0085.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0086.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0087.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0088.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0089.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0090.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0091.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0092.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0093.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0094.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0095.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0096.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0097.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0098.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0099.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0100.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0101.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0102.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0103.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0104.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0105.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0106.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0107.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0108.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0109.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0110.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0111.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0112.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0113.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0114.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0115.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0116.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0117.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0118.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0119.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0120.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0121.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0122.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0123.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0124.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0125.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0126.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0127.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0128.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0129.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0130.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0131.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0132.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0133.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0134.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0135.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0136.parquet"
    - "https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched/resolve/refs%2Fconvert%2Fparquet/default/train/0137.parquet"
---

# JuDDGES-pl

**JuDDGES-pl** is a unified Polish court judgment corpus for civil-law legal NLP,
accompanying the NeurIPS 2026 Evaluations & Datasets Track submission
*"JuDDGES-pl: Hundreds of Thousands of Structurally Enriched Polish Judgments
from Common and Administrative Courts"*.

The corpus combines two disjoint branches of the Polish judiciary into a single
Hugging Face dataset with two configs:

| Config | Source | Coverage | Rows |
|---|---|---|---|
| `pl-court` | [JuDDGES/pl-court-raw-enriched](https://huggingface.co/datasets/JuDDGES/pl-court-raw-enriched) | Common courts (Supreme Court, courts of appeal, regional and district courts) | 100K–1M |
| `pl-nsa` | [JuDDGES/pl-nsa-enriched](https://huggingface.co/datasets/JuDDGES/pl-nsa-enriched) | Administrative courts (Supreme Administrative Court and lower) | 1M–10M |

Each judgment includes:

- **Verbatim source content** (text, signature, court name, judges, dates, statute references) preserved from the official Polish court publication portals.
- **LLM-extracted analytical fields** generated by a uniform Google Gemini 2.5 Pro extraction pipeline: `factual_state`, `legal_state`, `extracted_summary`, `extracted_thesis`, `extracted_legal_bases`, `extracted_keywords`, `extracted_title`.

## ⚠️ Silver-label commitment

The LLM-extracted fields are **silver labels**: produced by a single model under
uniform prompting, **without per-document human validation at corpus scale**.

- **Use them for:** descriptive analysis, model pretraining, weak supervision, retrieval indexing.
- **Do not use them for:** benchmarking with held-out test sets, regulatory or
  individual legal decisions.
- **Always cite the model version** (Google Gemini 2.5 Pro) when reporting
  analyses derived from these fields, since LLM behavior shifts across releases.

A separate validated subset (`JuDDGES/pl-swiss-franc-loans`) provides gold
annotations for tasks requiring high-fidelity labels.

## Loading

```python
from datasets import load_dataset

common = load_dataset("JuDDGES/juddges-pl", "pl-court", split="train", streaming=True)
nsa    = load_dataset("JuDDGES/juddges-pl", "pl-nsa",   split="train", streaming=True)

for row in common.take(3):
    print(row["court_name"], row["signature"], row["extracted_summary"][:120])
```

## Licensing

- **Source judgments**: public-domain legal acts under Polish copyright law
  (Art. 4 pkt 2 of the Ustawa o prawie autorskim — *orzeczenia sądów* are
  *dokumenty urzędowe* and not subject to copyright).
- **Enrichment layer** (LLM-extracted analytical fields, structured metadata,
  Croissant): released under **CC BY 4.0**.

## Privacy

Source judgments are pseudonymized at the court level before publication
(personal names of natural persons → initials/codes; PESEL, addresses,
dates of birth redacted). Names of public officials, judges, and corporate
parties are typically retained as legally permitted. We perform no
additional de-anonymization. Users redistributing derivatives must comply
with GDPR/RODO and must not attempt re-identification.

## Maintenance

Maintained by the [JuDDGES organization](https://huggingface.co/JuDDGES) at
Wrocław University of Science and Technology. Versioned via Hugging Face
commits; dataset card updates announce material changes.

## Citation

```bibtex
@inproceedings{juddges-pl-2026,
  title     = {JuDDGES-pl: Hundreds of Thousands of Structurally Enriched Polish Judgments from Common and Administrative Courts},
  author    = {Augustyniak, {\L}ukasz and others},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year      = {2026}
}
```

## Croissant metadata

Croissant JSON-LD with full Responsible AI fields is released alongside the
paper (NeurIPS 2026 supplementary material).
