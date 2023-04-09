import Plot from "react-plotly.js";
import { Data } from "plotly.js";
import results from "../results.json";

const flatData = results.flat();

const lossDatas: Data[] = flatData.map((res) => {
  return {
    name: res.optimizer,
    x: res.results.map((r) => r.iteration / 300),
    y: res.results.map((r) => r.test.loss),
    type: "scatter",
  };
});

const percentages: Data[] = flatData.map((res) => {
  return {
    name: res.optimizer,
    x: res.results.map((r) => r.iteration / 300),
    y: res.results.map((r) => r.test.percentage),
    type: "scatter",
  };
});

function App() {
  return (
    <div className="App">
      <Plot
        data={lossDatas}
        layout={{
          width: 1600,
          height: 960,
          title: "Opetusjoukon virhe",
          xaxis: {
            title: "Epookki",
            dtick: 1,
          },
          yaxis: {
            title: "Virhe",
          },
        }}
      />
      <Plot
        data={percentages}
        layout={{
          width: 1600,
          height: 960,
          title: "Opetusjoukon luokittelutarkkuus",
          xaxis: {
            title: "Epookki",
            dtick: 1,
          },
          yaxis: {
            title: "Luokittelutarkkuus",
          },
        }}
      />

      <Plot
        data={[
          {
            type: "scatter",
            ...drawFunction(relu),
          },
        ]}
        layout={{
          width: 600,
          height: 600,
          title: "ReLu",
          xaxis: {
            range: [-2, 2],
            dtick: 0.5,
          },
          yaxis: {
            range: [-2, 2],
            dtick: 0.5,
          },
        }}
      />

      <Plot
        data={[
          {
            type: "scatter",
            ...drawFunction(leakyrelu),
          },
        ]}
        layout={{
          width: 600,
          height: 600,
          title: "LeakyReLu",
          xaxis: {
            range: [-2, 2],
            dtick: 0.5,
          },
          yaxis: {
            range: [-2, 2],
            dtick: 0.5,
          },
        }}
      />

      <Plot
        data={[
          {
            type: "scatter",
            ...drawFunction(TanhFn),
          },
        ]}
        layout={{
          width: 600,
          height: 600,
          title: "TanH",
          xaxis: {
            range: [-10, 10],
            dtick: 1,
          },
          yaxis: {
            range: [-1.5, 1.5],
            dtick: 0.1,
          },
        }}
      />

      <Plot
        data={[
          {
            type: "scatter",
            ...drawFunction(SigmoidFn),
          },
        ]}
        layout={{
          width: 600,
          height: 600,
          title: "Sigmoid",
          xaxis: {
            range: [-10, 10],
            dtick: 1,
          },
          yaxis: {
            range: [-1.5, 1.5],
            dtick: 0.1,
          },
        }}
      />
    </div>
  );
}

const drawFunction = (
  fn: (x: number) => number
): { x: number[]; y: number[] } => {
  const density = 300;
  const a = -10;
  const b = 10;
  const step = (b - a) / density;

  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < density; i += 1) {
    const point = a + step * i;
    x.push(point);
    y.push(fn(point));
  }
  return { x, y };
};

const leakyrelu = (x: number) => Math.max(0.2 * x, x);

const relu = (x: number) => Math.max(0, x);

function TanhFn(x: number) {
  return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
}

function SigmoidFn(x: number) {
  const divisor = 1 + Math.exp(-x);
  return 1 / divisor;
}

export default App;
