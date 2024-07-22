from typing import Any, List, Optional

from langflow.base.flow_processing.utils import build_data_from_run_outputs
from langflow.custom import Component
from langflow.inputs import MultilineInput, MessageTextInput, DropdownInput, NestedDictInput
from langflow.field_typing import NestedDict, Text
from langflow.graph.schema import RunOutputs
from langflow.schema import Data, dotdict


class RunFlowComponent(Component):
    display_name = "Run Flow"
    description = "A component to run a flow."
    name = "RunFlow"
    beta: bool = True

    inputs = [
        DropdownInput(
            name="flow_name",
            display_name="Flow Name",
            options=[],
            value="Streamlit",
            refresh_button=True,
            required=True,
        ),
        MultilineInput(
            name="input_value",
            display_name="Input Value"
        ),
        NestedDictInput(
            name="tweaks",
            display_name="Tweaks",
            info="Tweaks to apply to the flow.",
            value={"role": "user"}
        )
    ]

    outputs = [
        Output(display_name="Data", name="data", method="data_response"),
    ]

    def get_flow_names(self) -> List[str]:
        flow_data = self.list_flows()
        return [flow_data.name for flow_data in flow_data]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "flow_name":
            build_config["flow_name"]["options"] = self.get_flow_names()

        return build_config


    async def data_response(self) -> List[Data]: # , input_value: Text, flow_name: str, tweaks: NestedDict
        flow = filter(lambda x: x.name == self.flow_name, self.list_flows()).__next__()
        results: List[Optional[RunOutputs]] = await self.run_flow(
            inputs={"input_value": self.input_value}, flow_id=str(flow.id), flow_name=self.flow_name, tweaks=self.tweaks
        )
        if isinstance(results, list):
            data = []
            for result in results:
                if result:
                    data.extend(build_data_from_run_outputs(result))
        else:
            data = build_data_from_run_outputs()(results)

        self.status = data
        return data
