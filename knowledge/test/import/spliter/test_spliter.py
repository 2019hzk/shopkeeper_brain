from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "苹果的颜色是红色的[SEP]香蕉的颜色是黄色的[SEP]橘子的颜色是橙色的"

splitter_false = RecursiveCharacterTextSplitter(
    separators=["[SEP]"],
    chunk_size=10,
    chunk_overlap=0,
    keep_separator=False
)
# ['苹果的颜色是红色的', '[SEP]香蕉的颜色是黄色的', '[SEP]橘子的颜色是橙色的']
# ['苹果的颜色是红色的', '香蕉的颜色是黄色的', '橘子的颜色是橙色的']
print(splitter_false.split_text(''))