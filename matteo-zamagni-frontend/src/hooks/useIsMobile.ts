import { useGlobalContext } from "@/state/GlobalStore";

const MOBILE_WIDTH = 768;

export const useIsMobile = () => {
  const { screenDim } = useGlobalContext();

  return screenDim ? screenDim.x <= MOBILE_WIDTH : false;
};
