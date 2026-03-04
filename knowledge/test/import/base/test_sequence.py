from dataclasses import dataclass
from typing import Sequence
# tuple_container=(1,2,3)
#
#
# tuple_container[0]=10  # 不能修改任意位置的元素
# print(tuple_container)


# tuple_container1=([0,1,2],2,3)
# tuple_container1[0].append(4)
# print(tuple_container1)



@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    max_length: int = None

_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", max_length=65535),
    ScalarFieldSpec(field_name="title",max_length=65535),
)
_SCALAR_FIELDS[0].field_name="xxxxx"
print(_SCALAR_FIELDS)