import fs from "fs";
import { Result } from "./src/utils";

const data = [];
for (let i = 1; i <= 3; i += 1) {
  const file = fs.readFileSync(`results-${i}.json`).toString();
  data.push(...JSON.parse(file));
}
const oldResults = fs.readFileSync(`results-old.json`).toString();
data.push(...JSON.parse(oldResults).flat());

type Data = { optimizer: string; results: Result[]; took: number };
const results: Record<string, Data[]> = {};
for (const d of data) {
  if (!results[d.optimizer]) {
    results[d.optimizer] = [];
  }
  results[d.optimizer].push(d);
}
const averages: Data[] = [];
for (const [optimizer, datas] of Object.entries(results)) {
  const res: Data = { optimizer, took: 0, results: [] };
  const n = datas.length;
  for (let i = 0; i < n; i += 1) {
    const data = datas[i];
    if (res.took < 1000) {
      res.took += data.took / n;
    }
  }
  for (let i = 0; i < datas[0].results.length; i += 1) {
    const avg: Result = {
      iteration: datas[0].results[i].iteration,
      test: { loss: 0, percentage: 0 },
      train: { loss: 0, percentage: 0 },
    };
    for (let j = 0; j < n; j += 1) {
      const r = datas[j].results[i];
      avg.test.loss += r.test.loss / n;
      avg.test.percentage += r.test.percentage / n;
      avg.train.loss += r.train.loss / n;
      avg.train.percentage += r.train.percentage / n;
    }
    res.results.push(avg);
  }
  averages.push(res);
}

fs.writeFileSync("results.json", JSON.stringify(averages));
