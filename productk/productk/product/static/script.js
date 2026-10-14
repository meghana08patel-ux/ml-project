const $ = (id) => document.getElementById(id);


/* =========================================
   HELPER FUNCTIONS
========================================= */

function getArray(data, keys) {

    for (const key of keys) {

        if (Array.isArray(data?.[key])) {
            return data[key];
        }

    }

    return [];
}


function fillDropdown(element, values, placeholder) {

    element.innerHTML = "";

    element.add(new Option(placeholder, ""));

    values.forEach((item) => {

        if (item && typeof item === "object") {

            const value =
                item.value ??
                item.id ??
                item.store_nbr ??
                item.family ??
                item.name ??
                item.label;

            const label =
                item.label ??
                item.name ??
                item.family ??
                item.store_nbr ??
                value;

            if (value !== undefined) {

                element.add(
                    new Option(
                        String(label),
                        String(value)
                    )
                );

            }

        } else {

            element.add(
                new Option(
                    String(item),
                    String(item)
                )
            );

        }

    });

}


/* =========================================
   LOAD STORE AND PRODUCT OPTIONS
========================================= */

async function loadOptions() {

    try {

        const response = await fetch("/api/options");

        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                typeof data.detail === "string"
                    ? data.detail
                    : JSON.stringify(
                        data.detail || data
                    )
            );

        }


        /* LOAD STORES */

        fillDropdown(
            $("store"),
            getArray(
                data,
                [
                    "stores",
                    "store_nbrs",
                    "store_numbers"
                ]
            ),
            "Select a store"
        );


        /* LOAD PRODUCT FAMILIES */

        fillDropdown(
            $("family"),
            getArray(
                data,
                [
                    "families",
                    "product_families",
                    "family"
                ]
            ),
            "Select a product family"
        );


        /* CHECK OPTIONS */

        if (
            $("store").options.length <= 1 ||
            $("family").options.length <= 1
        ) {

            throw new Error(
                "Store or family options were missing from the API response."
            );

        }


        $("status").textContent =
            "Options loaded. Both Ridge and Lasso models will be used.";

    }


    catch (error) {

        $("status").className =
            "status error";


        $("status").textContent =
            "Could not load options: " +
            error.message;

    }

}


/* =========================================
   PREDICTION
========================================= */

$("forecastForm").addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        /* ---------------------------------
           BUTTON
        --------------------------------- */

        const button =
            $("predictButton");


        button.disabled = true;

        button.textContent =
            "Predicting demand...";


        /* ---------------------------------
           STATUS
        --------------------------------- */

        $("status").className =
            "status";

        $("status").textContent =
            "Generating Ridge + Lasso prediction...";


        /* ---------------------------------
           COLLECT USER INPUT
        --------------------------------- */

        const payload = {

            forecast_date:
                $("forecastDate").value,

            store_nbr:
                Number(
                    $("store").value
                ),

            family:
                $("family").value,

            onpromotion:
                Number(
                    $("promotion").value
                )

        };


        /* ---------------------------------
           AVAILABLE STOCK
        --------------------------------- */

        const availableStock =
            Number(
                $("stock").value
            );


        try {


            /* =================================
               CALL FASTAPI
            ================================= */

            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                payload
                            )
                    }
                );


            const data =
                await response.json();


            /* =================================
               CHECK API RESPONSE
            ================================= */

            if (!response.ok) {

                throw new Error(

                    typeof data.detail === "string"

                        ? data.detail

                        : JSON.stringify(
                            data.detail || data
                        )

                );

            }


            /* =================================
               GET RIDGE PREDICTION
            ================================= */

            const ridge =
                Number(
                    data.ridge_prediction
                );


            /* =================================
               GET LASSO PREDICTION
            ================================= */

            const lasso =
                Number(
                    data.lasso_prediction
                );


            /* =================================
               GET COMBINED PREDICTION
            ================================= */

            const combined =
                Number(
                    data.combined_prediction ??
                    data.predicted_sales ??
                    ((ridge + lasso) / 2)
                );


            /* =================================
               VALIDATE PREDICTIONS
            ================================= */

            if (
                !Number.isFinite(ridge) ||
                !Number.isFinite(lasso) ||
                !Number.isFinite(combined)
            ) {

                throw new Error(
                    "Invalid prediction returned by the API."
                );

            }


            /* =================================
               INVENTORY CALCULATION
            ================================= */

            const replenishment =
                Math.max(
                    0,
                    Math.ceil(
                        combined -
                        availableStock
                    )
                );


            /* =================================
               STOCK STATUS
            ================================= */

            let stockStatus;


            if (
                availableStock >= combined
            ) {

                stockStatus =
                    "✓ Stock Sufficient";

            } else {

                stockStatus =
                    "⚠ Stock Insufficient";

            }


            /* =================================
               SHOW RESULT SECTION
            ================================= */

            $("emptyResult").hidden =
                true;


            $("resultContent").hidden =
                false;


            /* =================================
               COMBINED PREDICTION
            ================================= */

            $("predictionValue").textContent =
                combined.toLocaleString(
                    undefined,
                    {
                        maximumFractionDigits: 0
                    }
                ) + " units";


            /* =================================
               INDIVIDUAL RIDGE PREDICTION
            ================================= */

            $("ridgeValue").textContent =
                ridge.toLocaleString(
                    undefined,
                    {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    }
                ) + " units";


            /* =================================
               INDIVIDUAL LASSO PREDICTION
            ================================= */

            $("lassoValue").textContent =
                lasso.toLocaleString(
                    undefined,
                    {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    }
                ) + " units";


            /* =================================
               INVENTORY — PREDICTED DEMAND
            ================================= */

            $("inventoryDemand").textContent =
                combined.toLocaleString(
                    undefined,
                    {
                        maximumFractionDigits: 0
                    }
                ) + " units";


            /* =================================
               INVENTORY — AVAILABLE STOCK
            ================================= */

            $("inventoryStock").textContent =
                availableStock.toLocaleString() +
                " units";


            /* =================================
               INVENTORY — REPLENISHMENT
            ================================= */

            $("replenishmentValue").textContent =
                replenishment.toLocaleString() +
                " units";


            /* =================================
               INVENTORY — STOCK STATUS
            ================================= */

            $("stockStatus").textContent =
                stockStatus;


            /* =================================
               FORECAST DATE
            ================================= */

            $("resultDate").textContent =
                payload.forecast_date;


            /* =================================
               STORE
            ================================= */

            $("resultStore").textContent =
                payload.store_nbr;


            /* =================================
               PRODUCT FAMILY
            ================================= */

            $("resultFamily").textContent =
                payload.family;


            /* =================================
               PROMOTION
            ================================= */

            $("resultPromotion").textContent =
                payload.onpromotion;


            /* =================================
               RESULT BADGE
            ================================= */

            $("resultBadge").textContent =
                "FORECAST READY";


            /* =================================
               RESULT EXPLANATION
            ================================= */

            $("resultDetails").textContent =

                "Ridge prediction: " +

                ridge.toLocaleString(
                    undefined,
                    {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    }
                ) +

                " units. Lasso prediction: " +

                lasso.toLocaleString(
                    undefined,
                    {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    }
                ) +

                " units. The final demand estimate is the average of both models.";


            /* =================================
               SUCCESS STATUS
            ================================= */

            $("status").className =
                "status";


            $("status").textContent =
                "Prediction generated successfully.";

        }


        /* =================================
           ERROR HANDLING
        ================================= */

        catch (error) {

            $("status").className =
                "status error";


            $("status").textContent =
                "Prediction failed: " +
                error.message;

        }


        /* =================================
           RESET BUTTON
        ================================= */

        finally {

            button.disabled =
                false;


            button.textContent =
                "Predict Demand";

        }

    }
);


/* =========================================
   INITIALIZE WEBSITE
========================================= */

loadOptions();