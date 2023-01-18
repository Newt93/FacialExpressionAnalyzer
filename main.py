from mediapipe.framework import calculator_pb2
from mediapipe.framework import packet
from mediapipe.python import util as mp_util

input_stream = 'input_video'
output_stream = 'output_video'

class FacialExpressionAnalyzer(mp_util.CalculatorBase):
    def __init__(self):
        super().__init__()
    def Open(self, calculator_context):
        self.facial_expression_model = calculator_context.solution.facial_expression_model
    def Process(self, calculator_context):
        input_packet = calculator_context.Inputs(input_stream).packet
        frame = input_packet.get_data()
        facial_expression = self.facial_expression_model.analyze(frame)
        output_packet = packet.Packet(facial_expression)
        calculator_context.Outputs(output_stream).packet = output_packet

pipeline = mp_util.create_pipeline(
    input_stream=input_stream,
    output_stream=output_stream,
    calculator=FacialExpressionAnalyzer(),
    solution='facial_expression_model')

mp_util.run_pipeline(pipeline, input_file='path/to/input_video.mp4', output_file='path/to/output_video.mp4')
