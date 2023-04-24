switch (process.argv[2]) {
  case "mnist": {
    const runMnist = require("./src/networks/mnist").default;
    runMnist();
    break;
  }
  case "xor": {
    const runXOR = require("./src/networks/xor").default;
    runXOR();
    break;
  }
  case "hotel": {
    const runHotel = require("./src/networks/hotel").default;
    runHotel();
    break;
  }
  case "fraud": {
    const runFraud = require("./src/networks/fraud").default;
    runFraud();
    break;
  }
  default: {
    throw new Error(`Invalid argument "${process.argv[2]}" supplied`);
  }
}
