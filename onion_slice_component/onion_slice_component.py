import streamlit as st
import streamlit.components.v1 as components
import json
import plotly
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def slice_viz_events(fig):
    try:
        # Convert the figure to JSON
        figJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Define the HTML template
        html_template = """
        <div id='plotly-chart'></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            const figJSON = %s;
            const plotlyChart = document.getElementById('plotly-chart');

            function updateShapes(eventdata) {
                const shapes = eventdata['shapes'] || plotlyChart.layout.shapes;
                if (shapes) {
                    const updatedShapes = shapes
                        .filter(shape => shape.type === 'line')
                        .map(shape => ({
                            x0: shape.x0,
                            y0: shape.y0,
                            x1: shape.x1,
                            y1: shape.y1
                        }));
                    
                    let outMessage = Object.assign({
                      isStreamlitMessage: true,
                      type: "streamlit:setComponentValue",
                    }, updatedShapes);
                
                    window.parent.postMessage(
                        outMessage,
                     '*');
                     console.log("Message sent to Streamlit", outMessage)

                }
            }

            Plotly.newPlot(plotlyChart, figJSON.data, figJSON.layout, {
                editable: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['hoverClosestCartesian', 'hoverCompareCartesian']
            }).then(() => {
                let isMouseDown = false;

                plotlyChart.on('plotly_relayout', updateShapes);

                plotlyChart.addEventListener('mousedown', () => { isMouseDown = true; });

                plotlyChart.addEventListener('mouseup', () => {
                    if (isMouseDown) {
                        isMouseDown = false;
                        updateShapes({shapes: plotlyChart.layout.shapes});
                    }
                });

                const deleteButton = document.querySelector('.modebar-btn[data-title="Remove active shape"]');
                if (deleteButton) {
                    deleteButton.addEventListener('click', () => {
                        setTimeout(() => {
                            updateShapes({shapes: plotlyChart.layout.shapes});
                        }, 100);
                    });
                }
            }).catch(err => {
                console.error('Error in Plotly:', err);
                window.parent.postMessage({
                    type: 'error',
                    error: err.toString()
                }, '*');
            });
        </script>
        """ % figJSON
        
        # Render the HTML template
        component_value = components.html(html_template, height=600)
        
        return component_value
        
    except Exception as e:
        logger.error(f"Error in slice_viz_events: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"An error occurred in the slice visualization: {str(e)}")
        return None
# Register the custom component
# components.declare_component("onion_slice_component", ".")
