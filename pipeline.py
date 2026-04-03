from clearml import PipelineController

# THIS IS JUST A SIMPLE SCRIPT WORK AS A PLACEHOLDER

def run_silent_script_pipeline():
    # 1. Initialize the Controller
    pipe = PipelineController(
        name="SilentScript_EndToEnd_Pipeline",
        project="ImConvo",
        version="1.0.0",
        add_pipeline_tags=True,
    )

    # 2. STEP 1: Preprocessing (The "Heaviest" node)
    # This will trigger scripts/preprocess.py
    pipe.add_step(
        name="data_preprocessing",
        base_task_project="ImConvo/Components",
        base_task_name="Preprocessing_Task",
        # ClearML will look for the task that ran preprocess.py
        # and clone it with these new parameters
    )

    # 3. STEP 2: Training (Dependent on Preprocessing)
    pipe.add_step(
        name="model_training",
        parents=["data_preprocessing"],
        base_task_project="ImConvo/Components",
        base_task_name="LipReadingCTC_Training",
        parameter_override={
            "Args/num_epochs": 100,
            "Args/batch_size": 48
        }
    )

    # 4. STEP 3: Evaluation (The "Gatekeeper")
    # This triggers test.py
    pipe.add_step(
        name="model_evaluation",
        parents=["model_training"],
        base_task_project="ImConvo/Components",
        base_task_name="Model_Evaluation_Task",
    )

    # 5. Execute on the Queue (Your "Real Worker" PC will pick this up)
    pipe.start(queue="default")

if __name__ == "__main__":
    run_silent_script_pipeline()
