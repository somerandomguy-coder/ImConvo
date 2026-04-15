from clearml import PipelineController

def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return

def run_silent_script_pipeline():
    # 1. Initialize the Controller
    pipe = PipelineController(
        name="ImConvo_EndToEnd_Pipeline",
        project="ImConvo",
        version="1.0.0",
        add_pipeline_tags=True,
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="download_data",
        base_task_id="3e57631724414b85ace469403b609a4d",
        post_execute_callback=post_execute_callback_example,
    )
    # 2. STEP 1: Preprocessing (The "Heaviest" node)
    # This will trigger scripts/preprocess.py
    pipe.add_step(
        name="preprocess_data",
        base_task_id="12ce84d486cf4ff494010dc2a7f48e7f",
        parents=[
            "download_data",
        ],
        parameter_override={
            "Args/parent": '${download_data.artifacts.taskID.url}' # 1 single parent
        },
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
        # ClearML will look for the task that ran preprocess.py
        # and clone it with these new parameters
    )

    # 3. STEP 2: Training (Dependent on Preprocessing)
    pipe.add_step(
        name="train_model",
        parents=[
            "preprocess_data",
        ],
        base_task_id="b0e56b32aa0f424eb112098fabac8238",
        parameter_override={
            "General/parents_ids": "${preprocess_data.artifacts.taskID.url}",  
        },
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    # 4. STEP 3: Evaluation (The "Gatekeeper")
    # This triggers test.py
    pipe.add_step(
        name="evaluate_model",
        parents=["train_model"],
        base_task_id="70327dd8bae54f16884aad603aabbf32",
        parameter_override={
            "Args/train_task_id":  "${train_model.artifacts.taskID.url}",
            "Args/preprocess_task_id": "${preprocess_data.artifacts.taskID.url}",
        },
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    # 5. Execute on the Queue (Your "Real Worker" PC will pick this up)
    pipe.start(queue="default")
    # pipe.start_locally(run_pipeline_steps_locally=False)


if __name__ == "__main__":
    run_silent_script_pipeline()
    print("Done")