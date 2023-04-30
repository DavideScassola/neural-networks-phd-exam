#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List

import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from .experiment import TrainingTrace


def training_trace_fig(trace: List[TrainingTrace]):
    task_x_traces = trace[0].x
    task_y_overlaps = tuple([elem.overlap for elem in trace])
    task_1_traces = tuple([elem.get_trace_task(0)[1] for elem in trace])
    task_2_traces = tuple([elem.get_trace_task(1)[1] for elem in trace])

    # ------------------------------------------------------------------------------
    custom_colorscale_task1 = sample_colorscale(
        colorscale="viridis", samplepoints=len(task_y_overlaps)
    )
    custom_colorscale_task2 = sample_colorscale(
        colorscale="plasma", samplepoints=len(task_y_overlaps)
    )

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Task 1", "Task 2"), shared_yaxes=True
    )
    for i in range(len(task_y_overlaps)):
        fig.add_trace(
            go.Scatter(
                x=task_x_traces,
                y=task_1_traces[i],
                mode="lines",
                name=f"overlap {task_y_overlaps[i]}",
                line=dict(color=custom_colorscale_task1[i]),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=task_x_traces,
                y=task_2_traces[i],
                mode="lines",
                name=f"overlap {task_y_overlaps[i]}",
                line=dict(color=custom_colorscale_task2[i]),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    for j in (1, 2):
        fig.update_xaxes(title_text="step", row=1, col=j)
    fig.update_yaxes(title_text="", type="log", row=1, col=2)
    fig.update_yaxes(title_text="loss", type="log", row=1, col=1)

    return fig
