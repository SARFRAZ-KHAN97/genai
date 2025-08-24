import tiktoken

enc= tiktoken.encoding_for_model("gpt-3.5-turbo")

text= "Hello, world! This is a test of the tokenization process."

tokens= enc.encode(text)
print("tokens: ", tokens)

tokens= [9906, 11, 1917, 0, 1115, 374, 264, 1296, 315, 279, 4037, 2065, 1920, 13]

decoded= enc.decode(tokens)
print("decoded: ", decoded)