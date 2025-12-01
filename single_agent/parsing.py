import time
from typing import Any, Dict
from jaxn import JSONParserHandler, StreamingJSONParser


class FinalReportHandler(JSONParserHandler):
    
    # def on_field_start(self, path: str, field_name: str) -> None:
    # # # #     # if field_name == "references":
    # # # #     #     header_level = path.count('/') + 2
    # # # #     #     print(f"\n\n{'#' * header_level} References\n")
    #     print(f"{path} - {field_name}")
        
    def on_field_end(self, path: str, field_name: str, value: str, parsed_value: Any = None) -> None:
        if field_name == "description":
            print(f'### Description')
            print(f'{value}')
        
        if field_name == "heading":
            print(f'\n\n## {value}\n')
        
        
         
    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        if field_name == "content":
            print(chunk, end="", flush=True)
    
    def on_array_item_end(self, path: str, field_name: str, item: Dict[str, Any] = None) -> None:
       if field_name == "references":
            print(f"\n ### References")
            if "episode_name" in item:
                print(f' - {item["episode_name"]} ({item["start_time"]} - {item["end_time"]})')
            else:
                author = item.get("author", "Unknown Author(s)")
                date_written = item.get("date_article_is_written", "n.d.")
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                print(f' - {author}. {date_written}. *{title}* {url}')




    
       
    
    
with open("messages.json", "r", encoding="utf-8") as f:
    data = f.read()

handler = FinalReportHandler()
parser = StreamingJSONParser(handler)

for i in range(0, len(data), 4):
    chunk = data[i:i+4]
    parser.parse_incremental(chunk)
    time.sleep(0.01)