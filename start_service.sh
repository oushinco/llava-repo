#!/bin/sh

# Start the controller
python -m llava.serve.controller --host 0.0.0.0 --port 10000 &

# Sleep for a moment to allow the controller to start
sleep 5


# Start the Gradio web server
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --port 7860 &


# Sleep for a moment to allow the controller to start
sleep 15

# Start the model worker
# python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-7b --load-8bit &

python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b &

# Keep the script running to keep the container alive
tail -f /dev/null
