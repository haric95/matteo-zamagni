import { useGlobalContext } from "@/state/GlobalStore";

const MOBILE_WIDTH = 768;
export const TABLET_WIDTH = 1024;

export const useIsMobile = (width?: number) => {
  const { screenDim } = useGlobalContext();

  return screenDim ? screenDim.x <= (width ?? MOBILE_WIDTH) : false;
};
