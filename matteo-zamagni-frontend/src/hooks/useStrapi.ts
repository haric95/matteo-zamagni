import { useCallback, useEffect, useState } from "react";

const STRAPI_URL =
  process.env.NEXT_PUBLIC_STRAPI_API_URL || "http://localhost:1337";

export type StrapiResponse<R> = {
  data: {
    id: number;
    attributes: {
      createdAt: string;
      publishedAt: string;
      updatedAt: string;
    } & R;
  };
  meta: any;
};

type StrapiImage = { url: string };

export type StrapiImageResponse = StrapiResponse<StrapiImage>;

export const useStrapi = <R = any, P = any>(
  path: string,
  // TODO: Make this type more specific
  strapiOptions: Record<string, string> = { populate: "*" }
) => {
  const [responseData, setResponseData] = useState<R | null>(null);
  const queryString = new URLSearchParams(strapiOptions).toString();

  const fetchData = useCallback(async () => {
    const url = `${STRAPI_URL}/api${path}?${queryString}`;
    const data = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
      },
    });
    if (data.status === 200) {
      const jsonData: StrapiResponse<R> = await data.json();
      setResponseData(jsonData.data.attributes);
    }
  }, [path, queryString]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return responseData;
};
