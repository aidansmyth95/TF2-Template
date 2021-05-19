source ./tf2_venv/bin/activate
echo "Launching Tensorboard..."
tensorboard --port=8080 --logdir=./saved_model