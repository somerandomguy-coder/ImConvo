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
        base_task_id="ad79af81ff2e44368fa8384a5d96577e",
        post_execute_callback=post_execute_callback_example,
    )
    # 2. STEP 1: Preprocessing (The "Heaviest" node)
    # This will trigger scripts/preprocess.py
    pipe.add_step(
        name="preprocess_data",
        base_task_id="8c253646b6f245dfadf520393058a34e",
        parents=[
            "download_data",
        ],
        parameter_override={
            "General/parent": "ad79af81ff2e44368fa8384a5d96577e", #str("${download_data.task_id}")  # 1 single parent
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
        base_task_id="95ebd286b4ba4a1ea26c9dbb8bad2628",
        parameter_override={
            "General/parents_ids": ["8c253646b6f245dfadf520393058a34e"],#[str("${preprocess_data.task_id}")],  # pass in as a list of parents
        },
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    # 4. STEP 3: Evaluation (The "Gatekeeper")
    # This triggers test.py
    pipe.add_step(
        name="evaluate_model",
        parents=["train_model"],
        base_task_id="0e6ad08bb596484d8480a401aaf7ecd0",
        parameter_override={
            "Args/train_task_id":  "95ebd286b4ba4a1ea26c9dbb8bad2628",#str("${train_model.task_id}"),
            "Args/preprocess_task_id": "8c253646b6f245dfadf520393058a34e", #str("${preprocess_data.task_id}"),
        },
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    # 5. Execute on the Queue (Your "Real Worker" PC will pick this up)
    pipe.start(queue="default")


if __name__ == "__main__":
    run_silent_script_pipeline()
    print("Done")