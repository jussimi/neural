import runXOR from "./src/networks/xor";
import runMnist from "./src/networks/mnist";
import runHotel from "./src/networks/hotel";

switch (process.argv[2]) {
  case "mnist": {
    runMnist();
    break;
  }
  case "xor": {
    runXOR();
    break;
  }
  case "hotel": {
    runHotel();
    break;
  }
  default: {
    throw new Error(`Invalid argument "${process.argv[2]}" supplied`);
  }
}
