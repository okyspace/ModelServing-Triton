IMAGE=nvcr.io/nvidia/tritonserver
VER=21.03-py3
HTTP=8000
GRPC=8001
METRIC=8002
GPU_MODE=true
GPUS=1
MODEL_REPO=/mnt/hdd/workspace_aieng/triton_space/samples/model_repo
POLL_RATE=10
CONTROL_MODE=poll
STRICT_MODE=true

if $GPU_MODE
then
	docker run -it --rm --gpus=$GPUS -p $HTTP:8000 -p $GRPC:8001 -p $METRIC:8002 -v $MODEL_REPO:/models $IMAGE:$VER tritonserver --model-repository=/models --model-control-mode=$CONTROL_MODE --repository-poll-secs=$POLL_RATE --strict-model-config=$STRICT_MODE --log-verbose=1

else 
	docker run -it --rm -p $HTTP:8000 -p $GRPC:8001 -p $METRIC:8002 -v $MODEL_REPO:/models $IMAGE:$VER tritonserver --model-repository=/models --model-control-mode=$CONTROL_MODE --repository-poll-secs=$POLL_RATE --strict-model-config=$STRICT_MODE --log-verbose=1
fi
