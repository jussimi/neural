type Prediction = {
  estimate: number[];
  output: number[];
};

export const binaryClassificationAccuracy = (
  points: Prediction[],
  cutoff = 0.5
) => {
  let correctCount = 0;

  for (const { estimate, output } of points) {
    const predicted = estimate[0] > cutoff ? 1 : 0;
    if (predicted === output[0]) {
      correctCount += 1;
    }
  }
  return correctCount / points.length;
};

export const classificationAccuracy = (points: Prediction[]) => {
  let correctCount = 0;

  for (const { estimate, output } of points) {
    const label = output.indexOf(1);
    const max = Math.max(...estimate);
    if (label === estimate.indexOf(max)) {
      correctCount += 1;
    }
  }
  return correctCount / points.length;
};

export const areaUnderCurve = (points: Prediction[], steps = 100) => {
  const curve: {
    cutoff: number;
    fpr: number;
    tpr: number;
    f: number;
    t: number;
  }[] = [];

  let positives = 0;
  let negatives = 0;

  for (const { output } of points) {
    if (output[0] === 1) {
      positives += 1;
    } else {
      negatives += 1;
    }
  }

  // Filter duplicate points.
  const seenPoints = new Map<number, true>();

  for (let i = 0; i <= steps; i += 1) {
    let falsePositives = 0;
    let truePositives = 0;

    const cutoff = i / steps;

    for (const { estimate, output } of points) {
      const prediction = estimate[0] > cutoff ? 1 : 0;
      const label = output[0];

      if (prediction === 1) {
        if (label === 1) {
          truePositives += 1;
        } else {
          falsePositives += 1;
        }
      }
    }

    if (!seenPoints.has(falsePositives)) {
      curve.push({
        fpr: falsePositives / negatives,
        tpr: truePositives / positives,
        f: falsePositives,
        t: truePositives,
        cutoff,
      });
      seenPoints.set(falsePositives, true);
    }
  }

  curve.sort((a, b) => a.fpr - b.fpr);

  // console.log(curve);

  let area = 0;

  for (let i = 0; i < curve.length - 1; i += 1) {
    const a = curve[i];
    const b = curve[i + 1];
    area += (b.fpr - a.fpr) * ((a.tpr + b.tpr) / 2);
  }

  return area;
};
