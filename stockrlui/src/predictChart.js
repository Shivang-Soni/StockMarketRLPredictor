import React from "react";
import {Line} from "react-chartjs-2";
import {
    Chart as ChartJS,
    LineElement,
    PointElement,
    LinearScale,
    CategoryScale,
    Tooltip,
    Legend,
    defaults,
    plugins,
    scales
} from "chart.js";

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend);

function  PredictChart({dates, predictions, actual_prices}) {
const data = {
labels: dates, 
datasets: [{
data: actual_prices,
label : "Actual Price",
borderColor : "green",
fill : false,
tension: 0
},
{
data: predictions,
label : "Predictions",
borderColor : "blue",
fill : true,
tension: 0
}]
};
const options = {
responsive : true,
plugins : {
legend : {
position : "top"
}
},
scales: {
    y: {
        beginAtZero:false,
        title: {
            display: true,
            text: "Price (in USD)"
        }
    }
}
};
return <Line data={data} options={options} />
}
export default PredictChart;