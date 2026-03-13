validate_item_names = ["商品A","商品B"]
# print(validate_item_names)
quoted = ", ".join(f'"{v}"' for v in validate_item_names)

print(f'item_name in {quoted}')
# print(f'item_name in {validate_item_names}')

# s='a,b,c'
# for i in s:
#     print(i)
# print(len(validate_item_names))
