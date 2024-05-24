import { useCallback, useEffect, useState } from "react";

const STRAPI_URL =
  process.env.NEXT_PUBLIC_STRAPI_API_URL || "http://localhost:1337";

type True = true;

type StrapiData<R> = {
  id: number;
  attributes: {
    createdAt: string;
    publishedAt: string;
    updatedAt: string;
  } & R;
};

type StrapiResponse<R = any, A = boolean> = {
  data: A extends True ? StrapiData<R>[] : StrapiData<R>;
  meta: any;
};

type StrapiImage = { url: string };

export type StrapiImageResponse = StrapiResponse<StrapiImage, false>;
export type StrapiImagesResponse = StrapiResponse<StrapiImage, true>;

export const useStrapi = <R = any, A = false>(
  path: string,
  // TODO: Make this type more specific
  strapiOptions: Record<string, string> = { populate: "*" }
) => {
  const [responseData, setResponseData] = useState<StrapiResponse<R, A> | null>(
    null
  );
  const queryString = new URLSearchParams(strapiOptions).toString();

  const fetchData = useCallback(async () => {
    const url = `${STRAPI_URL}/api${path}?${queryString}`;
    const data = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
      },
    });
    if (data.status === 200) {
      const jsonData: StrapiResponse<R, A> = await data.json();
      setResponseData(jsonData);
    }
  }, [path, queryString]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return responseData;
};
