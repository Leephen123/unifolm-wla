from types import SimpleNamespace
import torch

class CPUInputsCollator:
    def __init__(self,cfg):
        self.cfg=cfg
        self.interface=None
    def __call__(self,examples):
        from transformers import AutoProcessor
        from unifolm_wla.model.modules.vlm.QWen3 import _QWen3_VL_Interface
        if self.interface is None:
            from examples.unifolm_wla.pytree_cycle_fix import install_cycle_free_pytree
            install_cycle_free_pytree()
            processor=AutoProcessor.from_pretrained(self.cfg.framework.qwenvl.base_vlm)
            processor.tokenizer.padding_side='right'
            self.interface=SimpleNamespace(processor=processor,config=self.cfg,model=SimpleNamespace(device=torch.device('cpu')))
        inputs=_QWen3_VL_Interface.build_qwenvl_inputs(self.interface,
            images=[e['image'] for e in examples],instructions=[e['lang'] for e in examples],
            image_roles=[e['image_roles'] for e in examples],arm_types=[e['arm_type'] for e in examples])
        return {'examples':examples,'prepared_inputs':dict(inputs)}

def use_worker_inputs(model):
    current={}
    def prebuilt(*a,**kw):
        return current['inputs']
    model.qwen_vl_interface.build_qwenvl_inputs=prebuilt
    original_forward, original_predict = model.forward, model.predict_action
    def select(batch):
        inputs=batch['prepared_inputs']
        size=len(batch['examples'])
        assert inputs['input_ids'].shape[0] == size
        assert inputs['attention_mask'].shape == inputs['input_ids'].shape
        grid=inputs['image_grid_thw']
        assert grid.shape==(size*3,3) and bool((grid==grid.new_tensor([1,20,28])).all())
        assert 'assistant_mask' not in inputs
        current['inputs']=inputs
        return batch['examples']
    def forward(batch):
        return original_forward(select(batch))
    def predict_action(batch):
        return original_predict(select(batch))
    model.forward=forward
    model.predict_action=predict_action
