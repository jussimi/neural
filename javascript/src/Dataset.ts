export type DatasetItem = {
  input: number[];
  output: number[];
};

export class Dataset {
  items: DatasetItem[];

  batchSize: number;

  constructor(items: DatasetItem[], batchSize: number | undefined = undefined) {
    this.items = items;
    this.batchSize = batchSize || items.length;
  }

  generateBatch(): DatasetItem[] {
    const set: DatasetItem[] = [];
    const usedIndices = new Map<number, number>();

    while (set.length < this.batchSize) {
      inner: while (true) {
        const nextIndex = Math.floor(Math.random() * this.items.length);
        if (usedIndices.has(nextIndex)) {
          continue;
        }
        usedIndices.set(nextIndex, 1);
        const data = this.items[nextIndex];
        set.push(data);
        break inner;
      }
    }
    return set;
  }

  setBatchSize(size: number) {
    this.batchSize = size;
  }

  static createTrainAndTest(items: DatasetItem[], split: number) {
    const splitIndex = Math.ceil(split * items.length);
    const testData = items.slice(0, splitIndex);
    const trainData = items.slice(splitIndex);

    return {
      train: new Dataset(trainData),
      test: new Dataset(testData),
    };
  }
}
