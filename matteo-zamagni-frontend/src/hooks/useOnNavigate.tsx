import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

export const useOnNavigate = (callback: () => void) => {
  const pathname = usePathname();

  const [localPathname, setLocalPathname] = useState(pathname);

  useEffect(() => {}, [pathname]);

  useEffect(() => {
    if (pathname !== localPathname) {
      setLocalPathname(pathname);
      callback();
    }
  }, [pathname, callback, localPathname]);
};
