TEST_PROMPT_TEMPLATE = """

You are a helpful AI assistant and an expert extracting entities like brand and varieties from 'Product Descriptions'.

You MUST carefully and correctly extract entities only. 
A given `Product Description` may have multiple brand and variety entities.

Product Description: {product_desc}
AI:

"""
NER_PROMPT_TEMPLATE = """You are a helpful AI assistant and an expert named entity extractor designed to assist humans.
Given a `Product Description` from a human, your task is to extract the following named entities along with the corresponding span of text:

```
brand, variety
```

Please make sure you always give an output in the following JSON Array format: 
```json
[   
    {{{{
        "start_char": int /// start character of entity
        "end_char": int /// end character of entity
        "entity": str  /// entity name, one of [brand, variety]
        "value": str  /// entity value extracted from product description
    }}}},
    ....
]

Be sure you MUST carefully and correctly extract entities only. 
A given `Product Description` may have multiple brand and variety entities.
Also, In case of any entity not present in the description, you MUST return its value exactly as `None`.

Product Description: {product_desc}
AI:
"""