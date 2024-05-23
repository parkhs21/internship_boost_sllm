from src.main_block import MainBlock

main_block = MainBlock()
demo = main_block.block

if __name__=='__main__':
    demo.queue(default_concurrency_limit=5)
    demo.launch(server_name="0.0.0.0", server_port=7863)